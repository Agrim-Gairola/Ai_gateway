import os
import time
import logging
import asyncio
from sqlalchemy.orm import Session

from gateway.schemas import GenerateRequest, GenerateResponse, RoutingStrategy
from intelligence.classifier import QueryClassifier
from intelligence.complexity import ComplexityScorer
from intelligence.cost_estimator import CostEstimator
from intelligence.routing_engine import RoutingEngine
from intelligence.prompt_compressor import PromptCompressor
from intelligence.prompt_dna import PromptDNA
from intelligence.consensus_arbiter import ConsensusArbiter
from models.provider_registry import ProviderRegistry
from observability.database import RequestLog
from observability.policy_engine import PolicyEngine

logger = logging.getLogger("ai_gateway.router")


class GatewayRouter:

    def __init__(self):
        self.classifier = QueryClassifier()
        self.scorer = ComplexityScorer()
        self.cost_estimator = CostEstimator()
        self.routing_engine = RoutingEngine()
        self.provider_registry = ProviderRegistry()
        self.policy_engine = PolicyEngine()
        self.compressor = PromptCompressor()
        self.prompt_dna = PromptDNA()
        self.arbiter = ConsensusArbiter(shared_model=self.prompt_dna._model)

    async def route(self, request, team_id, request_id, db):

        compressed_prompt, compression_stats = self.compressor.compress(request.prompt)
        request.prompt = compressed_prompt
        logger.info(f"[{request_id}] Compression: {compression_stats}")

        cache_hit, cached_entry = self.prompt_dna.lookup(request.prompt)
        if cache_hit:
            logger.info(f"[{request_id}] DNA Cache HIT similarity={cached_entry['similarity_score']}")
            await self._log_request(db=db, request_id=request_id, team_id=team_id,
                model=cached_entry["model"] + "_cached", provider="prompt_dna_cache",
                query_type=cached_entry["query_type"], complexity=0.0, tokens_used=0, cost_usd=0.0, success=True)
            return GenerateResponse(
                request_id=request_id, content=cached_entry["response"],
                model_used=cached_entry["model"] + " (cached)", provider="prompt_dna_cache",
                query_type=cached_entry["query_type"], complexity_score=0.0,
                tokens_used=0, estimated_cost_usd=0.0,
                routing_reason=f"Prompt DNA cache hit — similarity {cached_entry['similarity_score']} — zero API cost",
                fallback_used=False, original_tokens=compression_stats["original_tokens"],
                compressed_tokens=compression_stats["compressed_tokens"],
                compression_ratio=compression_stats["ratio_achieved"],
                compression_savings_usd=0.0, compression_applied=compression_stats["compressed"])

        query_type = self.classifier.classify(request.prompt)
        logger.info(f"[{request_id}] Query classified as: {query_type}")

        complexity = self.scorer.score(request.prompt, query_type)
        logger.info(f"[{request_id}] Complexity score: {complexity:.2f}")

        selected_model, provider, routing_reason = self.routing_engine.select(
            query_type=query_type, complexity=complexity,
            strategy=request.strategy, preferred_model=request.preferred_model)

        estimated_cost = self.cost_estimator.estimate(
            model=selected_model, prompt=request.prompt, max_tokens=request.max_tokens)

        allowed, policy_reason = await self.policy_engine.check(
            team_id=team_id, model=selected_model, estimated_cost=estimated_cost, db=db)
        if not allowed:
            raise Exception(f"Policy violation: {policy_reason}")

        fallback_used = False

        if self.arbiter._available and request.strategy != "performance":
            try:
                cheap_models = [
                    ("llama-3.1-8b-instant", "groq"),
                    ("gemini-2.5-flash", "gemini"),
                ]
                results = await asyncio.gather(
                    *[self._execute_model(m, p, request) for m, p in cheap_models],
                    return_exceptions=True)

                responses = []
                for i, result in enumerate(results):
                    if not isinstance(result, Exception):
                        content, tokens, cost = result
                        responses.append({"model": cheap_models[i][0], "provider": cheap_models[i][1],
                                         "content": content, "tokens": tokens, "cost": cost})

                if len(responses) >= 2:
                    agreed, best, similarity = self.arbiter.check_agreement(responses)
                    if agreed:
                        content = best["content"]
                        tokens_used = best["tokens"]
                        actual_cost = best["cost"]
                        selected_model = best["model"]
                        provider = best["provider"]
                        routing_reason = f"SPD consensus (similarity={similarity}) — {best['model']} selected, frontier skipped"
                        logger.info(f"[{request_id}] SPD agreed")
                    else:
                        logger.info(f"[{request_id}] SPD disagreed — escalating to Llama 70B")
                        content, tokens_used, actual_cost = await self._execute_model("llama-3.3-70b-versatile", "groq", request)
                        selected_model = "llama-3.3-70b-versatile"
                        provider = "groq"
                        routing_reason = f"SPD no consensus (similarity={similarity}) — escalated to Llama 70B"
                elif len(responses) == 1:
                    content, tokens_used, actual_cost = responses[0]["content"], responses[0]["tokens"], responses[0]["cost"]
                    selected_model, provider = responses[0]["model"], responses[0]["provider"]
                else:
                    raise Exception("All SPD models failed")
            except Exception as e:
                logger.warning(f"[{request_id}] SPD failed: {e}")
                try:
                    content, tokens_used, actual_cost = await self._execute(provider, selected_model, request)
                except Exception:
                    fallback_used = True
                    selected_model, provider, content, tokens_used, actual_cost = await self._fallback(selected_model, request)
        else:
            try:
                content, tokens_used, actual_cost = await self._execute(provider, selected_model, request)
            except Exception as e:
                logger.warning(f"[{request_id}] Primary failed: {e}")
                fallback_used = True
                selected_model, provider, content, tokens_used, actual_cost = await self._fallback(selected_model, request)

        self.prompt_dna.store(prompt=request.prompt, response=content,
            model=selected_model, query_type=query_type, cost=actual_cost, tokens=tokens_used)

        compression_cost_saved = round(
            (compression_stats["original_tokens"] - compression_stats["compressed_tokens"]) * 0.0000025, 8)

        await self._log_request(db=db, request_id=request_id, team_id=team_id,
            model=selected_model, provider=provider, query_type=query_type,
            complexity=complexity, tokens_used=tokens_used, cost_usd=actual_cost, success=True)

        return GenerateResponse(
            request_id=request_id, content=content, model_used=selected_model,
            provider=provider, query_type=query_type, complexity_score=round(complexity, 3),
            tokens_used=tokens_used, estimated_cost_usd=round(actual_cost, 6),
            routing_reason=routing_reason, fallback_used=fallback_used,
            original_tokens=compression_stats["original_tokens"],
            compressed_tokens=compression_stats["compressed_tokens"],
            compression_ratio=compression_stats["ratio_achieved"],
            compression_savings_usd=compression_cost_saved,
            compression_applied=compression_stats["compressed"])

    async def _execute(self, provider, model, request):
        adapter = self.provider_registry.get(provider)
        return await adapter.generate(model=model, prompt=request.prompt,
            max_tokens=request.max_tokens, temperature=request.temperature)

    async def _execute_model(self, model, provider, request):
        adapter = self.provider_registry.get(provider)
        return await adapter.generate(model=model, prompt=request.prompt,
            max_tokens=request.max_tokens, temperature=request.temperature)

    async def _fallback(self, failed_model, request):
        adapter = self.provider_registry.get("groq")
        content, tokens, cost = await adapter.generate(
            model="llama-3.1-8b-instant", prompt=request.prompt,
            max_tokens=request.max_tokens, temperature=request.temperature)
        return "llama-3.1-8b-instant", "groq", content, tokens, cost

    async def _log_request(self, db, request_id, team_id, model, provider,
                           query_type, complexity, tokens_used, cost_usd, success):
        record = RequestLog(id=request_id, team_id=team_id, model=model, provider=provider,
            query_type=query_type, complexity_score=complexity,
            tokens_used=tokens_used, cost_usd=cost_usd, success=success)
        db.add(record)
        db.commit()
        logger.info(f"[{request_id}] Logged: {tokens_used} tokens, ${cost_usd:.6f}, model={model}")
