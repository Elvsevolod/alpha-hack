"""CSV parsing under the same monitored executor and original wall-clock deadline."""
import time
from ..util import digest


class BoundedCSVLoader:
    def __init__(self, executor, budget, workdir):
        self.executor, self.budget, self.workdir = executor, budget, workdir

    def __call__(self, path):
        from ..worker import read_frame
        if self.budget.remaining <= 0:
            raise TimeoutError("Original deadline exhausted before CSV loading")
        directory = self.workdir / "loading" / digest(str(path))
        result = self.executor.run(directory, {"mode": "load_csv", "input_path": str(path.resolve()),
            "read_dependencies": [str(path.resolve())]}, self.budget.remaining)
        if result["status"] != "completed":
            raise TimeoutError("CSV loading did not complete: " + result["status"])
        if self.budget.remaining <= 0:
            raise TimeoutError("Original deadline exhausted during CSV loading")
        frame = read_frame(directory / "loaded.json")
        if self.budget.remaining <= 0:
            raise TimeoutError("Original deadline exhausted while decoding loaded CSV")
        return frame.reset_index(drop=True)
