"""Reconstruct separate reviewable B blocks against the immutable A archive."""
import ast
import base64
import difflib
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]

def replace_node(source, final, name, method=False):
    def node(text):
        tree=ast.parse(text)
        nodes=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Controller').body if method else tree.body
        return next(n for n in nodes if getattr(n,'name',None)==name or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
    old,new=node(source),node(final)
    lines=source.splitlines(keepends=True);current=final.splitlines(keepends=True)
    return ''.join(lines[:old.lineno-1]+current[new.lineno-1:new.end_lineno]+lines[old.end_lineno:])


def run():
    output=ROOT/'outputs/qa_plan_implementation_A/B_blocks_v031';output.mkdir(exist_ok=False)
    with zipfile.ZipFile(ROOT/'dist/solution_v028.zip') as z:
        tree=ast.parse(z.read('agent_bundle.py'))
        encoded=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PAYLOAD' for t in n.targets))
    with zipfile.ZipFile(io.BytesIO(base64.b85decode(encoded))) as z:
        names=['automl_agent/'+name+'.py' for name in ('contracts','controller','data','evaluation','validation','cleanup')]
        previous={name:z.read(name).decode() for name in names}
    final={name:(ROOT/name).read_text() for name in names}
    manifest=[]
    for label in ('B1','B2','B3','B4'):
        current=previous.copy()
        if label=='B1':
            current['automl_agent/data.py']=final['automl_agent/data.py']
            current['automl_agent/contracts.py']=replace_node(previous['automl_agent/contracts.py'],final['automl_agent/contracts.py'],'TaskSpec')
            for method in ('__init__','_task','run'):
                current['automl_agent/controller.py']=replace_node(current['automl_agent/controller.py'],final['automl_agent/controller.py'],method,True)
        elif label=='B2':
            current['automl_agent/contracts.py']=replace_node(previous['automl_agent/contracts.py'],final['automl_agent/contracts.py'],'FinishArgs')
            current['automl_agent/controller.py']=replace_node(previous['automl_agent/controller.py'],final['automl_agent/controller.py'],'SYSTEM')
            for method in ('dispatch','report'):
                current['automl_agent/controller.py']=replace_node(current['automl_agent/controller.py'],final['automl_agent/controller.py'],method,True)
        elif label=='B3':
            for name in ('evaluation','validation'):current['automl_agent/'+name+'.py']=final['automl_agent/'+name+'.py']
            current['automl_agent/contracts.py']=replace_node(previous['automl_agent/contracts.py'],final['automl_agent/contracts.py'],'FeedbackPolicy')
            current['automl_agent/controller.py']=replace_node(previous['automl_agent/controller.py'],final['automl_agent/controller.py'],'revise_validation',True)
        else:
            current=final.copy();current['automl_agent/finalists.py']=(ROOT/'automl_agent/finalists.py').read_text()
        patch=''
        for name,source in current.items():
            compile(source,name,'exec')
            patch+=''.join(difflib.unified_diff(previous.get(name,'').splitlines(keepends=True),source.splitlines(keepends=True),fromfile='a/'+name,tofile='b/'+name))
        path=output/(label+'.patch');path.write_text(patch)
        manifest.append({'block':label,'patch':path.name,'sha256':hashlib.sha256(patch.encode()).hexdigest(),
            'sources':{name:hashlib.sha256(source.encode()).hexdigest() for name,source in current.items() if source!=previous.get(name)}})
        previous=current
    assert all(previous[name]==source for name,source in final.items())
    (output/'MANIFEST.json').write_text(json.dumps({'base_archive':'dist/solution_v028.zip','candidate':'dist/solution_v031.zip','blocks':manifest},indent=2)+'\n')
    print('Four separate blocks; intermediate sources compile; final sources match')

if __name__=='__main__':run()
