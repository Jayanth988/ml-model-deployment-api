FROM python:3.11-slim

ARG HTTP_PROXY
ARG HTTPS_PROXY

WORKDIR /app

COPY requirements.txt .

RUN HTTP_PROXY="$HTTP_PROXY" HTTPS_PROXY="$HTTPS_PROXY" pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]