from pathlib import Path

from tools.runtime_sources import production_agent_sources


def test_competition_sources_include_nested_executors_but_no_development_provider():
    root=Path(__file__).resolve().parents[1]
    paths=production_agent_sources(root)
    relative={p.relative_to(root/'automl_agent').as_posix() for p in paths}
    assert {'controller.py','qwen.py','eda/service.py','eda/error_analysis.py','eda/temporal.py'} <= relative
    assert not {'eda/deepseek.py','eda/agent.py','eda/__main__.py'} & relative
    assert not any('DEEPSEEK_API_KEY' in p.read_text() for p in paths)
    assert all(p.suffix=='.py' and p.resolve().is_relative_to(root/'automl_agent') for p in paths)
