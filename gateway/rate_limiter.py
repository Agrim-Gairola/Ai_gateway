import time, asyncio
from collections import defaultdict
from typing import Tuple, Dict

class InMemoryBucket:
    def __init__(self, rate, capacity):
        self.rate = rate
        self.capacity = capacity
        self.tokens = defaultdict(lambda: capacity)
        self.last_refill = defaultdict(time.time)
        self._lock = asyncio.Lock()

    async def consume(self, key, tokens=1):
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_refill[key]
            self.tokens[key] = min(self.capacity, self.tokens[key] + elapsed * self.rate)
            self.last_refill[key] = now
            if self.tokens[key] >= tokens:
                self.tokens[key] -= tokens
                return True, self.tokens[key]
            return False, self.tokens[key]

class RateLimiter:
    LIMITS = {"premium":{"rate":2.0,"capacity":20},"standard":{"rate":1.0,"capacity":10},"default":{"rate":0.5,"capacity":5}}
    def __init__(self): self._buckets = {}
    def _get_bucket(self, tier):
        if tier not in self._buckets:
            cfg = self.LIMITS.get(tier, self.LIMITS["default"])
            self._buckets[tier] = InMemoryBucket(cfg["rate"], cfg["capacity"])
        return self._buckets[tier]
    async def check(self, team_id, tier="standard"):
        bucket = self._get_bucket(tier)
        allowed, remaining = await bucket.consume(team_id)
        if not allowed:
            return False, f"Rate limit exceeded. Tokens remaining: {remaining:.2f}"
        return True, f"OK ({remaining:.1f} tokens remaining)"
