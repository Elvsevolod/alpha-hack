"""Check direct production and candidate imports against organizer requirements."""
import ast
from pathlib import Path
import re
import sys
import json

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from automl_agent.custom import ALLOWED_IMPORTS
from automl_agent.util import atomic_json,file_hash

ROOT=Path(__file__).resolve().parents[1]
MODULE_DISTRIBUTIONS={'sklearn':'scikit-learn','imblearn':'imbalanced-learn','feature_engine':'feature-engine',
    'category_encoders':'category-encoders','yaml':'PyYAML','dotenv':'python-dotenv','autogluon':'autogluon.tabular','orjson':'orjson'}


def audit():
    requirements=ROOT/'data_for_participants/requirements.txt'
    pins={}
    for line in requirements.read_text().splitlines():
        match=re.fullmatch(r'([A-Za-z0-9_.-]+)(?:\[([^]]+)\])?==([^\s#]+)',line.strip())
        if match:pins[match[1]]={'version':match[3],'extras':match[2]}
    violations=[];imports=[]
    paths=[ROOT/'solution.py',*sorted((ROOT/'automl_agent').glob('*.py')),
        *[p for p in sorted((ROOT/'context_compressor').glob('*.py')) if p.name != 'verify.py'],
        *[ROOT/'ml_knowledge'/name for name in ('__init__.py','core.py','lexical.py','runtime.py','memory.py','context.py')]]
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.Import):modules=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom):
                if node.level:continue
                modules=[node.module or '']
            else:continue
            for module in modules:
                top=module.split('.')[0]
                if top in sys.stdlib_module_names or top in {'automl_agent','context_compressor','ml_knowledge','ml_knowledge_bundle'}:continue
                distribution=MODULE_DISTRIBUTIONS.get(top,top)
                allowed=distribution in pins and (top!='autogluon' or module.startswith('autogluon.tabular'))
                entry={'file':str(path.relative_to(ROOT)),'line':node.lineno,'module':module,'distribution':distribution,'allowed':allowed}
                imports.append(entry)
                if not allowed:violations.append(entry)
    candidate_violations=[]
    for module in ALLOWED_IMPORTS:
        distribution=MODULE_DISTRIBUTIONS.get(module,module)
        if module not in sys.stdlib_module_names and distribution not in pins:candidate_violations.append(module)
    report={'requirements_sha256':file_hash(requirements),'allowed_distributions':len(pins),'pins':pins,
        'direct_imports':imports,'production_violations':violations,'candidate_allowlist_violations':candidate_violations,
        'passed':not violations and not candidate_violations,
        'runtime_installation':False,'worker_process_creation':'blocked_by_import_policy_and_audit_hook',
        'worker_network':'blocked_by_audit_hook_and_offline_container_test',
        'trust_model':'limited_trust; native-code sandbox escape is not claimed prevented'}
    atomic_json(ROOT/'outputs/dependency_audit.json',report)
    print(json.dumps({'passed':report['passed'],'allowed_distributions':len(pins),'direct_imports':len(imports),'violations':len(violations)}))
    return report


if __name__=='__main__':raise SystemExit(0 if audit()['passed'] else 1)
