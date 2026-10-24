"""Development-only Qwen EDA runner; account checks stay outside competition runtime."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import httpx
from dotenv import dotenv_values

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from automl_agent.eda.agent import EDAAgent
from automl_agent.eda.deepseek import DeepSeekClient
from automl_agent.eda.planning import generation_inputs
from automl_agent.eda.trace import sanitized,event
from automl_agent.qwen import QwenOutcomeUnknown,QwenUnavailable,GENERATION
from automl_agent.util import atomic_json,digest

spec=importlib.util.spec_from_file_location('alpha_limits',ROOT/'skills/alpha-api-limits/scripts/check_limits.py')
limits=importlib.util.module_from_spec(spec);spec.loader.exec_module(limits)


class GatewayLedger:
    """Track reported response charges and account deltas, without USD assumptions.

    The gateway exposes no pre-request tariff. The cap is a stop threshold checked
    between sequential requests, not a server-enforced limit on an in-flight request.
    Unknown outcomes stop the whole series; headers/credentials are never persisted.
    """
    def __init__(self,path,key,before,run_id,cap=5):
        self.path,self.key,self.before,self.run_id,self.cap=Path(path),key,before,run_id,cap
        self.state=json.loads(self.path.read_text()) if self.path.exists() else {'requests':[],'cap_units':cap}
        if self.state['cap_units']!=cap:raise ValueError('Saved gateway threshold cannot change')

    def checkpoint(self):
        after=limits.snapshot(limits.fetch_info(self.key),self.key,'checkpoint')
        delta=limits.compare(self.before,after).get('increase')
        if delta is None or after.get('budget_exhausted') or after['limits'].get('blocked'):
            raise QwenUnavailable('Gateway accounting unknown or blocked')
        self.state['account_increase']=delta
        self.state['last_checkpoint']=after
        atomic_json(self.path,self.state)
        return delta

    def reserve(self,payload):
        if any(x['state']=='pending' for x in self.state['requests']):
            raise QwenOutcomeUnknown('Previous generation has an unknown outcome; series stopped')
        spent=self.checkpoint()
        reported=sum(x.get('units',0) for x in self.state['requests'])
        if reported>=self.cap:raise QwenUnavailable('Gateway series stop threshold reached')
        self.state['requests'].append({'run_id':self.run_id,'state':'pending','request_hash':digest(payload)})
        atomic_json(self.path,self.state)
        return len(self.state['requests'])-1

    def settle(self,ticket,cost,usage):
        try:cost=float(cost)
        except (ValueError,TypeError):raise QwenOutcomeUnknown('Gateway response charge missing') from None
        if not math.isfinite(cost) or cost<0 or not isinstance(usage,dict):
            raise QwenOutcomeUnknown('Invalid charge or usage')
        self.state['requests'][ticket].update(state='settled',units=cost,usage=usage)
        atomic_json(self.path,self.state)
        self.checkpoint()

    def summary(self):
        rows=[x for x in self.state['requests'] if x['run_id']==self.run_id]
        return {'gateway_units':sum(x.get('units',0) for x in rows),'requests':len(rows),
            'series_account_increase':self.state.get('account_increase'),
            'stop_threshold_units':self.cap,'threshold_enforcement':'between_requests',
            'outcome_unknown':any(x['state']=='pending' for x in self.state['requests'])}


class GatewayClient(DeepSeekClient):
    """Reuse the existing decoder and wide-contract encoding, with Qwen payloads."""
    def __init__(self,journal,budget,ledger,config,transport='native',client=None):
        self.journal,self.budget,self.ledger,self.transport=journal,budget,ledger,transport
        self.key,self.base_url,self.model=config['key'],config['base_url'],config['model']
        self.client=client or httpx.Client(timeout=60,follow_redirects=False)
        self.reserve_seconds=0;self.sent_requests=journal.get('eda_gateway_requests',0);self.provider_mode='live_qwen'
        journal.put('development_provider',{'provider':'organizer_qwen','model':self.model,
            'max_output_tokens':8192,'thinking':'medium','configuration':'development_only'})

    def _request(self,messages,tools=None,*,prepared=None):
        if not self.configured:raise QwenUnavailable('Organizer configuration missing')
        allowed=getattr(self,'allowed_tools',None)
        messages,tools,allowed=generation_inputs(messages,tools,allowed,self.budget)
        if allowed:self.allowed_tools=allowed
        if allowed and not tools:
            content=json.loads(messages[-1]['content'])
            if 'tool_argument_schemas' in content:
                names=[t['function']['name'] for t in allowed]
                content['tool_argument_schemas']={k:v for k,v in content['tool_argument_schemas'].items() if k in names}
                content['command_schema']['properties']['tool']['enum']=names
                from automl_agent.util import canonical
                messages=[*messages[:-1],{**messages[-1],'content':canonical(content)}]
        payload={'model':self.model,'messages':messages,'max_tokens':8192,'stream':False,
            **GENERATION,'reasoning':{'enabled':True,'effort':'medium'}}
        if tools:payload.update(tools=allowed or tools,tool_choice='required',parallel_tool_calls=False)
        else:payload['response_format']={'type':'json_object'}
        # Local call/time gates run before any sent generation.
        self.budget.claim_call()
        ticket=self.ledger.reserve(payload)
        atomic_json(self.journal.directory/'provider'/f'request_{ticket}.json',sanitized(payload,self.key))
        event(self.journal.directory,'provider_request',ticket=ticket,provider='organizer_qwen',request_hash=digest(payload))
        try:
            self.sent_requests+=1
            self.journal.put('eda_gateway_requests',self.sent_requests)
            r=self.client.post(self.base_url.rstrip('/')+'/chat/completions',json=payload,
                headers={'Authorization':'Bearer '+self.key},timeout=min(120,max(.1,self.budget.remaining)))
            r.raise_for_status();body=r.json()
            charge=r.headers.get('x-litellm-response-cost',(body.get('usage') or {}).get('cost'))
            self.ledger.settle(ticket,charge,body.get('usage'))
            message=body['choices'][0]['message']
            atomic_json(self.journal.directory/'provider'/f'response_{ticket}.json',
                sanitized({'message':message,'usage':body.get('usage'),'charge_units':charge,
                    'finish_reason':body['choices'][0].get('finish_reason')},self.key))
            event(self.journal.directory,'provider_response',ticket=ticket,usage=body.get('usage'),cost=self.ledger.summary())
            if body['choices'][0].get('finish_reason')=='length':
                raise QwenUnavailable('Truncated generation; retained usage, no automatic retry')
            self.journal.put("eda_received_response_call",self.budget.counter("llm_calls"))
            return message
        except (QwenUnavailable,QwenOutcomeUnknown):raise
        except Exception as e:
            event(self.journal.directory,'provider_unknown',ticket=ticket,error_type=type(e).__name__)
            raise QwenOutcomeUnknown('Organizer generation unknown: '+type(e).__name__) from None


def run_single(task, output, cap=5, policy='evidence', seconds=600):
    """Generic repository CLI route: organizer Qwen only; no dataset-specific paths."""
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    if (output/'run').exists():
        raise ValueError('Choose a new output for live acceptance; historical runs are not silently resumed')
    config=dotenv_values(ROOT/'.env')
    cfg={'key':os.getenv('OPENAI_API_KEY') or config.get('OPENAI_API_KEY'),
        'base_url':os.getenv('OPENAI_BASE_URL') or config.get('OPENAI_BASE_URL'),
        'model':os.getenv('OPENAI_MODEL') or config.get('OPENAI_MODEL') or 'qwen/qwen3.8-27b'}
    before=limits.snapshot(limits.fetch_info(cfg['key']),cfg['key'],'before')
    atomic_json(output/'api_limits_before.json',before)
    if before['budget_exhausted'] or before['limits']['blocked']:raise ValueError('Gateway unavailable')
    def factory(journal,budget,transport):
        ledger=GatewayLedger(output/'gateway_costs.json',cfg['key'],before,'single',cap)
        return ledger,GatewayClient(journal,budget,ledger,cfg,transport)
    agent=None
    try:
        agent=EDAAgent(task,output/'run',None,provider_factory=factory,policy=policy,seconds=seconds)
        return agent.autonomous()
    finally:
        if agent:agent.close()
        try:
            after=limits.snapshot(limits.fetch_info(cfg['key']),cfg['key'],'after');after['comparison']=limits.compare(before,after)
        except Exception as error:
            after={'phase':'after','remaining_budget':None,'error':type(error).__name__}
        atomic_json(output/'api_limits_after.json',after)


def run_series(tasks,output,cap=5,policies=('evidence',),names=('adult','bank','bike','har')):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/'runs').exists():raise ValueError('Series already started; inspect results before rerunning')
    config=dotenv_values(ROOT/'.env')
    cfg={'key':os.getenv('OPENAI_API_KEY') or config.get('OPENAI_API_KEY'),
        'base_url':os.getenv('OPENAI_BASE_URL') or config.get('OPENAI_BASE_URL'),
        'model':os.getenv('OPENAI_MODEL') or config.get('OPENAI_MODEL') or 'qwen/qwen3.8-27b'}
    before=limits.snapshot(limits.fetch_info(cfg['key']),cfg['key'],'before')
    atomic_json(output/'api_limits_before.json',before)
    if before['budget_exhausted'] or before['limits']['blocked']:raise ValueError('Gateway blocked or exhausted')
    results={};(output/'runs').mkdir()
    try:
        for policy in policies:
            for name in names:
                run_id=policy+'_'+name
                def factory(journal,budget,transport):
                    ledger=GatewayLedger(output/'gateway_costs.json',cfg['key'],before,run_id,cap)
                    return ledger,GatewayClient(journal,budget,ledger,cfg,transport)
                agent=EDAAgent(Path(tasks)/name/'task.md',output/'runs'/run_id,None,policy=policy,
                    provider_factory=factory,seconds=600)
                try:result=agent.autonomous()
                finally:agent.close()
                results[run_id]=result;atomic_json(output/'series.json',results)
                print(json.dumps({'run':run_id,'state':result['state'],'fits':result['model_fits'],
                    'calls':result['api_calls'],'cost':result['cost'],'stop':result['stop_reason']}),flush=True)
                if result['cost']['outcome_unknown'] or 'Gateway series stop threshold' in result['stop_reason']:
                    return results
        return results
    finally:
        after=limits.snapshot(limits.fetch_info(cfg['key']),cfg['key'],'after')
        after['comparison']=limits.compare(before,after)
        atomic_json(output/'api_limits_after.json',after)
        print(json.dumps({'after_remaining':after['remaining_budget'],'increase':after['comparison'].get('increase')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('tasks',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stop-units',type=float,default=5);p.add_argument('--policies',nargs='+',choices=['free','evidence','alternatives'],default=['evidence'])
    p.add_argument('--datasets',nargs='+',choices=['adult','bank','bike','har'],default=['adult','bank','bike','har'])
    a=p.parse_args();run_series(a.tasks,a.output,a.stop_units,a.policies,a.datasets)
