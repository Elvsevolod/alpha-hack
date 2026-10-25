"""A valid archive or offline test run never promotes predictive quality."""
import argparse
import json
import math
from pathlib import Path
import statistics


def evaluate(manifest, results):
    required={'protocol_hash','versions','tasks','seeds','resources','metric_directions','evaluation_independence'}
    if not required<=set(manifest) or len(manifest['versions'])!=2:
        raise ValueError('A frozen paired protocol is required')
    baseline,candidate=manifest['versions']
    expected={(version,task,seed) for version in manifest['versions'] for task in manifest['tasks'] for seed in manifest['seeds']}
    observed={}
    for row in results:
        key=(row['version'],row['task'],row['seed'])
        if key not in expected or key in observed:
            raise ValueError('Unexpected or duplicate paired result')
        if row.get('protocol_hash')!=manifest['protocol_hash'] or row.get('resources')!=manifest['resources']:
            raise ValueError('Result protocol/resources differ from the preregistered pair')
        observed[key]=row
    complete=set(observed)==expected
    valid=lambda row: row.get('valid_csv') is True and row.get('control_status')=='completed' and row.get('refit_status')=='completed' and row.get('publication_scope')=='full_train_refit' and type(row.get('raw_metric')) in (int,float) and math.isfinite(row['raw_metric'])
    valid_rate=sum(valid(row) for row in observed.values())/max(1,len(expected))
    tasks={}
    for task in manifest['tasks']:
        maximize=manifest['metric_directions'][task]=='maximize'
        pairs=[]
        for seed in manifest['seeds']:
            a,b=observed.get((baseline,task,seed)),observed.get((candidate,task,seed))
            if a and b and valid(a) and valid(b):
                pairs.append({'seed':seed,'baseline':a['raw_metric'],'candidate':b['raw_metric'],
                    'signed_improvement':(b['raw_metric']-a['raw_metric'])*(1 if maximize else -1)})
        metrics=[row['candidate'] for row in pairs]
        tasks[task]={'pairs':pairs,'candidate_worst':(min(metrics) if maximize else max(metrics)) if metrics else None,
            'mean_signed_improvement':statistics.mean(row['signed_improvement'] for row in pairs) if pairs else None,
            'uncertainty':'single_trajectory' if len(pairs)<3 else 'empirical_repeats_no_significance_claim'}
    independent=manifest['evaluation_independence']=='fresh_frozen_audit'
    real_agent_evidence=bool(observed) and all(row.get('real_qwen') is True for row in observed.values())
    eligible=complete and valid_rate==1 and independent and real_agent_evidence and len(set(manifest['seeds']))>=3 and all(
        row['pairs'] and min(pair['signed_improvement'] for pair in row['pairs'])>=0 and row['mean_signed_improvement']>0 for row in tasks.values())
    return {'status':'eligible_for_quality_review' if eligible else 'quality_advantage_unestablished',
        'complete':complete,'valid_submission_rate':valid_rate,'tasks':tasks,
        'independent_audit':independent,'automatic_promotion':False,
        'live_qwen_runs':sum(row.get('real_qwen') is True for row in observed.values()),
        'offline_checks_are_quality_evidence':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('manifest',type=Path);parser.add_argument('results',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=evaluate(json.loads(args.manifest.read_text()),json.loads(args.results.read_text()))
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False))
