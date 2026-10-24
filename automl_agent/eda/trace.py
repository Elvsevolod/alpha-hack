"""Small local event log for explaining actions, evidence and stop conditions."""
from datetime import datetime, timezone
import json


def event(directory,phase,**fields):
    directory.mkdir(parents=True,exist_ok=True)
    record={'time':datetime.now(timezone.utc).isoformat(),'phase':phase,**fields}
    with (directory/'events.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')


def sanitized(value,secret):
    if isinstance(value,str):return value.replace(secret,'[REDACTED]') if secret else value
    if isinstance(value,list):return [sanitized(item,secret) for item in value]
    if isinstance(value,dict):return {sanitized(k,secret):sanitized(v,secret) for k,v in value.items()}
    return value
