import pytest

from automl_agent.custom import validate_code


@pytest.mark.parametrize('code',[
    'from sklearn.datasets import load_iris\ndef build_pipeline(context): return load_iris()',
    'from sklearn import datasets\ndef build_pipeline(context): return datasets.load_iris()',
    'from numpy import load\ndef build_pipeline(context): return load("weights")',
    'import torch\ndef build_pipeline(context): return torch.hub.load("anything")',
    'import numpy as np\ndef build_pipeline(context): return np.ctypeslib.load_library("anything",".")',
    'import requests\ndef build_pipeline(context): return None',
    'import networkx\ndef build_pipeline(context): return None',
    'import subprocess\ndef build_pipeline(context): return subprocess.run(["pip","install","anything"])',
    'import pandas as pd\ndef analyze(X,y,context):\n dependency=pd.compat._optional.import_optional_dependency("packaging")\n return {"unlisted_library_loaded":str(dependency)}\ndef build_pipeline(context):return None',
])
def test_disallowed_direct_imports_data_sources_and_installer_routes(code):
    with pytest.raises(ValueError):validate_code(code,'pipeline')


def test_json_feature_parsing_remains_available():
    validate_code('import json\ndef build_pipeline(context):\n return json.loads("{}")','pipeline')


def test_runtime_blocks_cached_dynamic_dependency_without_breaking_library_internals(tmp_path):
    import subprocess,sys,json
    candidate=tmp_path/'candidate.py'
    candidate.write_text('import pandas as pd\ndef analyze():\n return pd.compat._optional.import_optional_dependency("packaging")\n')
    source='''
from pathlib import Path
import importlib.util
import pandas as pd
import packaging
from automl_agent.dependency_policy import install_dependency_guard
root=Path(__import__('sys').argv[1]); path=root/'candidate.py'
install_dependency_guard(path,root)
spec=importlib.util.spec_from_file_location('candidate_module',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.analyze()
raise RuntimeError('Unlisted module escaped the runtime guard')
'''
    result=subprocess.run([sys.executable,'-c',source,str(tmp_path)],capture_output=True,text=True,timeout=15)
    assert result.returncode==78,(result.stdout,result.stderr)
    outcome=json.loads((tmp_path/'worker_result.json').read_text())
    assert outcome['status']=='unsupported_mode' and 'packaging' in outcome['error']
