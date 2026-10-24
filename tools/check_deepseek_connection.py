"""Read-only DeepSeek authentication/model/balance check; never sends a generation."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import httpx
from dotenv import dotenv_values
from automl_agent.util import atomic_json


def check(output, client=None):
    config=dotenv_values(Path.cwd()/'.env')
    key=os.getenv('DEEPSEEK_API_KEY') or config.get('DEEPSEEK_API_KEY')
    base=(os.getenv('DEEPSEEK_BASE_URL') or config.get('DEEPSEEK_BASE_URL') or 'https://api.deepseek.com').rstrip('/')
    model=os.getenv('DEEPSEEK_MODEL') or config.get('DEEPSEEK_MODEL') or 'deepseek-flash'
    result={'checked_at':datetime.now(timezone.utc).isoformat(),'generation_requests':0,'key_present':bool(key),
        'models_http_status':None,'balance_http_status':None,'requested_model_available':None,'balance_available':None}
    parsed=urlsplit(base)
    if not key:
        result['diagnosis']='missing_key'
    elif parsed.scheme!='https' or parsed.hostname!='api.deepseek.com' or parsed.username or parsed.password:
        result['diagnosis']='unverified_endpoint_no_credentials_sent'
    else:
        own=client is None
        client=client or httpx.Client(timeout=20,follow_redirects=False)
        try:
            for name,path in [('models','/models'),('balance','/user/balance')]:
                response=client.get(base+path,headers={'Authorization':'Bearer '+key})
                result[name+'_http_status']=response.status_code
                if response.status_code==200:
                    body=response.json()
                    if name=='models':
                        result['requested_model_available']=any(row.get('id')==model for row in body.get('data',[]))
                        result['model_count']=len(body.get('data',[]))
                    else:result['balance_available']=body.get('is_available') if type(body.get('is_available')) is bool else None
            codes=[result['models_http_status'],result['balance_http_status']]
            result['diagnosis']=('authentication_failed' if 401 in codes else 'insufficient_balance' if 402 in codes or result['balance_available'] is False
                else 'model_unavailable' if result['requested_model_available'] is False else 'read_only_checks_ok' if codes==[200,200] else 'http_failure')
        except (httpx.HTTPError,ValueError,TypeError) as error:
            result['diagnosis']='transport_or_response_failure';result['error_type']=type(error).__name__
        finally:
            if own:client.close()
    atomic_json(Path(output),result)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(check(args.output),ensure_ascii=False))
