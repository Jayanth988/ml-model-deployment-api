import asyncio
import time
from statistics import mean
from urllib.request import Request, urlopen
import json

URL = "http://localhost:8000/api/v1/predict"
API_KEY = "dev-secret-key-change-this"

PAYLOAD = json.dumps({
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}).encode("utf-8")


def send_request():
    request = Request(
        URL,
        data=PAYLOAD,
        headers={
            "Content-Type": "application/json",
            "X-API-Key": API_KEY,
        },
        method="POST",
    )

    start = time.perf_counter()

    try:
        with urlopen(request, timeout=10) as response:
            response.read()
            latency = time.perf_counter() - start
            return True, latency

    except Exception:
        latency = time.perf_counter() - start
        return False, latency


async def main():
    total_requests = 100

    start = time.perf_counter()

    results = await asyncio.gather(
        *[
            asyncio.to_thread(send_request)
            for _ in range(total_requests)
        ]
    )

    total_duration = time.perf_counter() - start

    successful = [
        latency for success, latency in results if success
    ]

    failed = [
        latency for success, latency in results if not success
    ]

    print("\n===== LOAD TEST RESULTS =====")
    print(f"Total requests : {total_requests}")
    print(f"Successful     : {len(successful)}")
    print(f"Failed         : {len(failed)}")
    print(f"Total duration : {total_duration:.4f} seconds")

    if successful:
        print(f"Average latency: {mean(successful):.4f} seconds")
        print(f"Minimum latency: {min(successful):.4f} seconds")
        print(f"Maximum latency: {max(successful):.4f} seconds")


if __name__ == "__main__":
    asyncio.run(main())