import time, logging
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("ai_gateway")
logging.basicConfig(level=logging.INFO)

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        logger.info(str(request.url.path))
        return response

class MetricsMiddleware(BaseHTTPMiddleware):
    _c = {"requests_total": 0, "requests_success": 0}
    async def dispatch(self, request, call_next):
        MetricsMiddleware._c["requests_total"] += 1
        response = await call_next(request)
        if response.status_code < 400:
            MetricsMiddleware._c["requests_success"] += 1
        return response

    @classmethod
    def get_metrics(cls):
        return cls._c
