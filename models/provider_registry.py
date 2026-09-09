import os
import asyncio
import logging
import warnings
from abc import ABC, abstractmethod
from typing import Tuple, AsyncGenerator
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the repository root, wherever the repo happens to live.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

logger = logging.getLogger("ai_gateway.providers")


class BaseProvider(ABC):
    @abstractmethod
    async def generate(self, model, prompt, max_tokens, temperature) -> Tuple[str, int, float]:
        pass

    async def stream(self, model, prompt, max_tokens, temperature) -> AsyncGenerator[str, None]:
        content, tokens, cost = await self.generate(model, prompt, max_tokens, temperature)
        yield content


class GroqProvider(BaseProvider):
    PRICING = {
        "llama-3.1-8b-instant":    (0.00005, 0.00008),
        "llama-3.3-70b-versatile": (0.00059, 0.00079),
    }

    async def generate(self, model, prompt, max_tokens, temperature) -> Tuple[str, int, float]:
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            logger.warning("GROQ_API_KEY not found, using mock")
            return await MockProvider("groq").generate(model, prompt, max_tokens, temperature)
        try:
            from groq import Groq
            client = Groq(api_key=api_key)

            def _call():
                return client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

            response = await asyncio.get_event_loop().run_in_executor(None, _call)
            content = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            tokens = input_tokens + output_tokens
            in_price, out_price = self.PRICING.get(model, (0.0001, 0.0001))
            cost = (input_tokens / 1000 * in_price) + (output_tokens / 1000 * out_price)
            return content, tokens, round(cost, 8)
        except Exception as e:
            logger.warning(f"Groq failed: {e}, using mock")
            return await MockProvider("groq").generate(model, prompt, max_tokens, temperature)

    async def stream(self, model, prompt, max_tokens, temperature) -> AsyncGenerator[str, None]:
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            content, _, _ = await MockProvider("groq").generate(model, prompt, max_tokens, temperature)
            yield content
            return
        try:
            from groq import Groq
            client = Groq(api_key=api_key)

            def _call():
                return client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                )

            stream = await asyncio.get_event_loop().run_in_executor(None, _call)
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.warning(f"Groq stream failed: {e}")
            content, _, _ = await MockProvider("groq").generate(model, prompt, max_tokens, temperature)
            yield content


class GeminiProvider(BaseProvider):
    PRICING = {
        "gemini-2.5-flash":        (0.0002,  0.0008),
        "gemini-2.0-flash":        (0.0001,  0.0004),
        "gemini-1.5-flash":        (0.000075, 0.0003),
        "gemini-1.5-flash-latest": (0.000075, 0.0003),
    }

    async def generate(self, model, prompt, max_tokens, temperature) -> Tuple[str, int, float]:
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            logger.warning("GEMINI_API_KEY not found, using mock")
            return await MockProvider("gemini").generate(model, prompt, max_tokens, temperature)
        try:
            warnings.filterwarnings("ignore")
            import google.generativeai as genai
            genai.configure(api_key=api_key)

            def _call():
                m = genai.GenerativeModel(model)
                return m.generate_content(
                    prompt,
                    generation_config={"max_output_tokens": max_tokens, "temperature": temperature},
                )

            response = await asyncio.get_event_loop().run_in_executor(None, _call)
            content = response.text
            try:
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count
            except Exception:
                input_tokens = len(prompt.split())
                output_tokens = len(content.split())
            tokens = input_tokens + output_tokens
            in_price, out_price = self.PRICING.get(model, (0.0001, 0.0003))
            cost = (input_tokens / 1000 * in_price) + (output_tokens / 1000 * out_price)
            return content, tokens, round(cost, 8)
        except Exception as e:
            logger.warning(f"Gemini failed: {e}, using mock")
            return await MockProvider("gemini").generate(model, prompt, max_tokens, temperature)


class MockProvider(BaseProvider):
    def __init__(self, provider_name="mock"):
        self.provider_name = provider_name

    async def generate(self, model, prompt, max_tokens, temperature) -> Tuple[str, int, float]:
        await asyncio.sleep(0.1)
        content = f"[MOCK {model}] Response to: {prompt[:60]}..."
        tokens = len(prompt.split()) + 80
        cost = tokens * 0.000001
        return content, tokens, round(cost, 8)

    async def stream(self, model, prompt, max_tokens, temperature) -> AsyncGenerator[str, None]:
        words = f"[MOCK {model}] Response to: {prompt[:60]}...".split()
        for word in words:
            await asyncio.sleep(0.05)
            yield word + " "


class ProviderRegistry:
    def __init__(self):
        self._providers = {
            "groq":   GroqProvider(),
            "gemini": GeminiProvider(),
            "mock":   MockProvider(),
        }

    def get(self, provider: str) -> BaseProvider:
        p = self._providers.get(provider)
        if not p:
            logger.warning(f"Unknown provider {provider}, falling back to groq")
            return self._providers["groq"]
        return p
