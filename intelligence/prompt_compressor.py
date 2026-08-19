import logging
from typing import Tuple

logger = logging.getLogger("ai_gateway.compressor")


class PromptCompressor:
    MIN_WORDS_TO_COMPRESS = 25

    def __init__(self):
        self._compressor = None
        self._available = False
        self._try_load()

    def _try_load(self):
        try:
            import torch
            from llmlingua import PromptCompressor as LC
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._compressor = LC(
                model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                use_llmlingua2=True,
                device_map=device,
            )
            logger.info(f"LLMLingua-2 loaded on {device}")
            self._available = True
        except Exception as e:
            logger.warning(f"LLMLingua-2 not available: {e}")
            self._available = False

    def compress(self, prompt: str, ratio: float = 0.5) -> Tuple[str, dict]:
        word_count = len(prompt.split())

        if not self._available or word_count < self.MIN_WORDS_TO_COMPRESS:
            return prompt, {
                "original_tokens": word_count,
                "compressed_tokens": word_count,
                "ratio_achieved": 1.0,
                "compressed": False,
                "skipped_reason": "too_short" if word_count < self.MIN_WORDS_TO_COMPRESS else "unavailable",
            }

        try:
            result = self._compressor.compress_prompt(
                prompt,
                rate=ratio,
                force_tokens=["\n", "?", ".", "!"],
            )
            compressed = result["compressed_prompt"]
            compressed_words = len(compressed.split())
            ratio_achieved = compressed_words / word_count

            logger.info(f"Compressed {word_count} -> {compressed_words} words ({ratio_achieved:.2f}x)")

            return compressed, {
                "original_tokens": word_count,
                "compressed_tokens": compressed_words,
                "ratio_achieved": round(ratio_achieved, 3),
                "compressed": True,
                "skipped_reason": None,
            }

        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return prompt, {
                "original_tokens": word_count,
                "compressed_tokens": word_count,
                "ratio_achieved": 1.0,
                "compressed": False,
                "skipped_reason": "error",
            }