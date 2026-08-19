import logging
import numpy as np
from typing import Tuple, List

logger = logging.getLogger("ai_gateway.arbiter")


class ConsensusArbiter:

    AGREEMENT_THRESHOLD = 0.75

    def __init__(self, shared_model=None):
        self._model = None
        self._available = False
        self._try_load(shared_model)

    def _try_load(self, shared_model=None):
        try:
            import faiss
            if shared_model is not None:
                self._model = shared_model
                logger.info("Consensus Arbiter initialized with shared model")
            else:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Consensus Arbiter initialized with own model")
            self._available = True
        except Exception as e:
            logger.warning(f"Consensus Arbiter not available: {e}")
            self._available = False

    def _embed(self, text: str) -> np.ndarray:
        embedding = self._model.encode([text[:500]])[0]
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def check_agreement(self, responses: List[dict]) -> Tuple[bool, dict, float]:
        if not self._available or len(responses) < 2:
            return False, responses[0] if responses else {}, 0.0
        try:
            embeddings = [self._embed(r["content"]) for r in responses]
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = float(np.dot(embeddings[i], embeddings[j]))
                    similarities.append(sim)
                    logger.info(
                        f"Similarity {responses[i]['model']} <-> "
                        f"{responses[j]['model']}: {sim:.4f}"
                    )
            avg_similarity = sum(similarities) / len(similarities)
            agreed = avg_similarity >= self.AGREEMENT_THRESHOLD
            if agreed:
                best = min(responses, key=lambda r: r["cost"])
                logger.info(f"CONSENSUS REACHED similarity={avg_similarity:.4f} winner={best['model']}")
            else:
                best = responses[0]
                logger.info(f"NO CONSENSUS similarity={avg_similarity:.4f} escalating to frontier model")
            return agreed, best, round(avg_similarity, 4)
        except Exception as e:
            logger.warning(f"Arbiter check failed: {e}")
            return False, responses[0], 0.0
