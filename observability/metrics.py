class _NoOp:
    def labels(self, **kwargs): return self
    def inc(self, *a, **kw): pass
    def observe(self, *a, **kw): pass
    def set(self, *a, **kw): pass

metrics_registry = None
REQUEST_COUNT = _NoOp()
REQUEST_LATENCY = _NoOp()
COST_COUNTER = _NoOp()
TOKEN_COUNTER = _NoOp()
ACTIVE_REQUESTS = _NoOp()
