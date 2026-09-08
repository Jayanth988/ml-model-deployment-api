FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# 0.0.0.0 binds Uvicorn to all network interfaces inside the container.
# This allows Docker to forward published host traffic to the API.
# 127.0.0.1 would bind only to the container's loopback interface,
# so the API would not be reachable through Docker's published port.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]