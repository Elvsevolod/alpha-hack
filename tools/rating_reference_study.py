"""EPA-motivated target parameterization, evaluated on forward periods."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.further_reference_study import configurations
from tools.literature_reference_study import run_case, write_report


def study(output, seconds):
    root = Path(output).resolve(); root.mkdir(parents=True, exist_ok=False)
    lgb, cb = configurations(False)[:2]
    configs = [lgb, cb] + [dict(lgb, name=f'gamma_rating_{shape}', alternative='rating', rating_shape=shape)
                           for shape in [4., 16.]]
    run_case(root, 'regression', time.monotonic() + seconds, config_override=configs,
        baseline_override=39.03614722027966, reference_pair=['current_lgb', 'current_cb'])
    write_report(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=1200)
    args = parser.parse_args()
    study(args.output, args.seconds)
