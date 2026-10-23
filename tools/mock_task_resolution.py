"""Semantic reply oracle for synthetic technical fixtures; never packaged."""
import json


def fixture_resolution(body):
    request=json.loads(body['messages'][-1]['content'])
    if not isinstance(request,dict) or 'sources' not in request or 'request_id' not in request:
        return None
    catalog=json.loads(request['sources']['inventory']['text'])
    task={'contract_version':2,'file_roles':{name:name for name in ('train.csv','test.csv','sample_submission.csv','data_dictionary.csv') if name in catalog},
        'kind':'regression','targets':['y'],'id_columns':['id'],'feature_columns':['x','cat'],'categorical_columns':['cat'],
        'metric':{'name':'rmse'},'output':{'columns':['id','y'],'prediction_type':'values'},'source_hash':request['source_hash']}
    proposal={'request_id':request['request_id'],'source_hash':request['source_hash'],'status':'proposed','task_spec':task,
        'evidence':{field:['task','inventory'] for field in task}}
    return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps(proposal)}}],
        'usage':{'prompt_tokens':10,'completion_tokens':20}}
