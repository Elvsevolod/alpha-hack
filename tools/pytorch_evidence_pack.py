"""Archive measured evidence without trained weights or private runtime config."""
from pathlib import Path
import hashlib,json,shutil,xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def run():
    target=ROOT/'research/pytorch_author_recipe_20261026/evidence'
    target.mkdir(parents=True,exist_ok=True)
    source=ROOT/'outputs/realmlp_author_equal_20261026'
    manifest=[]
    def copy(path,destination):
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,destination)
        manifest.append(dict(source=str(path.relative_to(ROOT)),artifact=str(destination.relative_to(ROOT)),
            sha256=hashlib.sha256(destination.read_bytes()).hexdigest()))
    for name in ['freeze.json','execution_environment.json','prediction_audit.json','har_control_admission.json']:
        if (source/name).is_file():copy(source/name,target/name)
    rows=[]
    for directory in sorted(source.glob('*/*/*')):
        if not directory.is_dir():continue
        relative=directory.relative_to(source)
        if not (directory/'cost.json').exists():continue
        cost=json.loads((directory/'cost.json').read_text())
        result=json.loads((directory/'result.json').read_text()) if (directory/'result.json').exists() else {}
        row=dict(phase=relative.parts[0],task=relative.parts[1],arm=relative.parts[2],
            status=result.get('status',cost.get('stop') or 'failure'),score=result.get('score'),
            raw_score=result.get('raw_score'),weight=result.get('weight'),cost=cost,
            base_fits=result.get('base_fits'),postprocessing_fits=result.get('postprocessing_fits'),
            fit_ledger=[{k:v for k,v in f.items() if k!='curve'} for f in result.get('fits',[])])
        rows.append(row)
        for name in ['result.json','cost.json','failure.json','charged_protocol.json']:
            if (directory/name).exists():copy(directory/name,target/relative/name)
        # Compressed arrays preserve the exact held-out predictions and row lineage.
        arrays={p.stem:np.load(p) for p in directory.glob('*.npy')}
        if (directory/'charged_splits.npz').exists():
            with np.load(directory/'charged_splits.npz') as split:
                arrays.update({'split_'+k:split[k] for k in split.files})
        if arrays:
            dest=target/relative/'predictions_and_rows.npz';np.savez_compressed(dest,**arrays)
            manifest.append(dict(source=str(relative),artifact=str(dest.relative_to(ROOT)),sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    comparisons=[]
    for phase,task in sorted({(r['phase'],r['task']) for r in rows}):
        arms={r['arm']:r for r in rows if r['phase']==phase and r['task']==task}
        if 'trees' not in arms or arms['trees']['score'] is None:continue
        base=arms['trees']['score']
        for arm,r in arms.items():
            if r['score'] is None:continue
            comparisons.append(dict(phase=phase,task=task,arm=arm,score=r['score'],
                gain_vs_trees=(base-r['score']) if task in {'regression','bike'} else (r['score']-base),
                wall_ratio_vs_trees=r['cost']['wall_seconds']/arms['trees']['cost']['wall_seconds']))
    (target/'summary.json').write_text(json.dumps(dict(rows=rows,comparisons=comparisons,api_calls=0,
        quality_scope='reused internal evaluation; seed replicas are not independent controls'),indent=2)+'\n')
    # Earlier screening and the separately frozen transfer check are historical context.
    legacy=ROOT/'research/pytorch_agent_20261026/evidence';legacy.mkdir(exist_ok=True)
    for name in ['pytorch_gate_final.json','pytorch_control_freeze.json','pytorch_host.json','pytorch_attention_cost.json']:
        p=ROOT/'outputs'/name
        if p.exists():copy(p,legacy/name)
    for task in ['adult','bank','bike']:
        for name in ['results.json','protocol.json','splits.npz']:
            p=ROOT/'outputs/pytorch_study_control42'/task/name
            if p.exists():copy(p,legacy/'control'/task/name)
    audit=ROOT/'outputs/pytorch_reference_verification/verification.json'
    if audit.exists():copy(audit,target/'historical_reference_verification.json')
    checks=[]
    for pattern in ['*pytorch*xml','realmlp*xml','agent_path_audit*xml']:
        for p in sorted((ROOT/'outputs').glob(pattern)):
            xml=ET.parse(p).getroot();suites=[xml] if xml.tag=='testsuite' else list(xml)
            checks.append(dict(source=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                counts={key:sum(int(s.get(key,0)) for s in suites) for key in ['tests','failures','errors','skipped']},
                failures=[dict(test=case.get('name'),kind=child.tag) for case in xml.iter('testcase') for child in case if child.tag in {'failure','error'}]))
    (target/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(completed_arms=len(rows),comparisons=comparisons),indent=2))


if __name__=='__main__':run()
