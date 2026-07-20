FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Flat module layout (no package dir) - see main.py/field_zones.py/ndvi.py/schemas.py/
# geometry_utils.py/config.py, all at repo root.
COPY *.py ./

RUN useradd --system --create-home --home-dir /app appuser \
    && mkdir -p /app/logs \
    && chown -R appuser:appuser /app
USER appuser

# SH_CLIENT_ID/SH_CLIENT_SECRET (see config.py's Settings) must be set as real Railway env vars -
# .env is intentionally excluded by .dockerignore, never baked into the image.

# Railway assigns PORT dynamically; ${PORT:-8000} falls back to 8000 for `docker run` outside
# Railway. Keep this in sync with the fixed private-network port documented in kret's
# NDVI_API_URL comment (application.properties) - set PORT=8000 explicitly as this service's
# Railway variable so that other services can reliably reach lopata.railway.internal:8000.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
