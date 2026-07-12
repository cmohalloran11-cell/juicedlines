# JUICED — persistent FastAPI server (needs a long-running process for the
# background refresh loop + in-memory cache; NOT a serverless/Vercel app).
FROM python:3.12-slim

WORKDIR /app

# Install deps first so the layer caches across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hosts (Render/Railway/Fly) inject the port via $PORT; default for plain `docker run`.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
