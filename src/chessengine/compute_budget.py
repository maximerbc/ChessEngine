"""Conservative persisted allowance for active local study wall time."""
from contextlib import AbstractContextManager
from pathlib import Path
import time
from .common import read_json, write_json


class BudgetExceeded(RuntimeError):
    pass


class ComputeBudget(AbstractContextManager):
    def __init__(self, path, seconds):
        self.path, self.allowance = Path(path), seconds
        self.state = read_json(path) if self.path.exists() else dict(allowance=seconds, consumed=0., running_since=None)
        if self.state['allowance'] != seconds:
            raise ValueError('Compute allowance mismatch')
        # A crashed process cannot prove its stop time; charge the whole unclosed interval.
        if self.state['running_since'] is not None:
            self.state['consumed'] += max(0., time.time()-self.state['running_since'])
        self.started = time.perf_counter()
        self.state['running_since'] = time.time()
        write_json(self.path, self.state)

    @property
    def remaining(self):
        return self.allowance-self.state['consumed']-(time.perf_counter()-self.started)

    def check(self):
        if self.remaining <= 1:
            raise BudgetExceeded('Local study compute allowance exhausted; progress preserved')

    def checkpoint(self):
        self.state['consumed'] += time.perf_counter()-self.started
        self.started = time.perf_counter()
        self.state['running_since'] = time.time()
        write_json(self.path, self.state)

    def __exit__(self, *args):
        self.checkpoint()
        self.state['running_since'] = None
        write_json(self.path, self.state)
        return False
