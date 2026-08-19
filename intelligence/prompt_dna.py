import hashlib
import logging
import numpy as np
from typing import Optional, Tuple
from datetime import datetime
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
logger = logging.getLogger("ai_gateway.prompt_dna")


class PromptDNA:

    SIMILARITY_THRESHOLD = 0.85
    MAX_CACHE_SIZE = 1000

    def __init__(self):
        self._model = None
        self._index = None
        self._cache = []
        self._available = False
        self._hits = 0
        self._misses = 0
        self._try_load()

    def _try_load(self):
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            dim = 384
            self._index = faiss.IndexFlatIP(dim)
            self._available = True
            logger.info("Prompt DNA initialized")
        except Exception as e:
            logger.warning(f"Prompt DNA not available: {e}")
            self._available = False

    def _embed(self, text: str) -> np.ndarray:
        embedding = self._model.encode([text])[0]
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(np.float32)

    def _fingerprint(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def lookup(self, prompt: str) -> Tuple[bool, Optional[dict]]:
        if not self._available or len(self._cache) == 0:
            self._misses += 1
            return False, None
        try:
            embedding = self._embed(prompt)
            query = embedding.reshape(1, -1)
            distances, indices = self._index.search(query, 1)
            similarity = float(distances[0][0])
            best_idx = int(indices[0][0])
            if similarity >= self.SIMILARITY_THRESHOLD and best_idx < len(self._cache):
                cached = self._cache[best_idx]
                self._hits += 1
                logger.info(f"Cache HIT similarity={similarity:.4f}")
                return True, {**cached, "similarity_score": round(similarity, 4), "cache_hit": True}
            self._misses += 1
            return False, None
        except Exception as e:
            logger.warning(f"Cache lookup failed: {e}")
            self._misses += 1
            return False, None

    def store(self, prompt: str, response: str, model: str,
              query_type: str, cost: float, tokens: int):
        if not self._available:
            return
        try:
            if len(self._cache) >= self.MAX_CACHE_SIZE:
                self._cache.pop(0)
                self._index.reset()
                if self._cache:
                    embeddings = np.array([self._embed(e["prompt"]) for e in self._cache])
                    self._index.add(embeddings)
            embedding = self._embed(prompt)
            self._index.add(embedding.reshape(1, -1))
            self._cache.append({
                "prompt": prompt,
                "response": response,
                "model": model,
                "query_type": query_type,
                "cost": cost,
                "tokens": tokens,
                "fingerprint": self._fingerprint(prompt),
                "stored_at": datetime.utcnow().isoformat(),
            })
            logger.info(f"Stored fingerprint={self._fingerprint(prompt)} cache_size={len(self._cache)}")
        except Exception as e:
            logger.warning(f"Cache store failed: {e}")

    def get_stats(self) -> dict:
        total = self._hits + self._misses
        hit_rate = round(self._hits / total * 100, 1) if total > 0 else 0
        return {
            "cache_size": len(self._cache),
            "total_lookups": total,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_pct": hit_rate,
            "available": self._available,
        }