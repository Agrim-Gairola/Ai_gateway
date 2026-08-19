# Multi-stage build for lean production image
FROM python:3.12-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source
COPY gateway/ ./gateway/
COPY intelligence/ ./intelligence/
COPY models/ ./models/
COPY observability/ ./observability/

# Non-root user for security
RUN useradd -r -s /bin/false appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
