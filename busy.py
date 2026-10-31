"""Which NpuStudio jobs are running right now (for the Restart button's 'still running' check)."""
import threading
_lock = threading.Lock(); _jobs = {}
def mark(name, on):
    with _lock: _jobs[name] = max(0, _jobs.get(name, 0) + (1 if on else -1))
def running():
    with _lock: return [k for k, v in _jobs.items() if v > 0]
