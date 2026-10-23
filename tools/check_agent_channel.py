"""One bounded model decision using a prepared technical CSV fixture."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
from automl_agent.budget import Limits
from automl_agent.controller import Controller, SYSTEM
from automl_agent.util import atomic_json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('task', type=Path)
    parser.add_argument('--transport', choices=['auto', 'native', 'json'], default='json')
    args = parser.parse_args(); load_dotenv(Path(__file__).resolve().parents[1] / '.env', override=False)
    controller = Controller(args.task, Limits(seconds=300, cpu=1, max_calls=1), run_id='bounded_channel', transport=args.transport)
    started = time.time()
    try:
        controller.prepare()
        try:
            command, _ = controller.qwen.next_command(SYSTEM, controller.context_memory.snapshot(SYSTEM))
            result = {'tool': command.tool, 'transport': controller.qwen.transport}
        except (ValueError, RuntimeError, TimeoutError) as error:
            result = {'error': type(error).__name__ + ': ' + str(error)}
        result.update(kind='channel_probe_not_behavioral_acceptance', seconds=time.time()-started,
            llm_seconds=controller.journal.get('llm_seconds'), token_usage=controller.journal.get('token_usage'),
            configuration=controller.journal.get('qwen_configuration'))
        atomic_json(args.task.parent / 'CHANNEL_PROBE.json', result)
        print(json.dumps(result, ensure_ascii=False))
    finally:
        controller.close()
