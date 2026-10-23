import asyncio
import base64
import csv
import json
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image as PILImage
from openai import AzureOpenAI
from azure.storage.blob import BlobServiceClient

from current_affairs.config import (
    CA_GEN_AZURE_OPENAI_API_KEY,
    CA_GEN_AZURE_OPENAI_API_VERSION,
    CA_GEN_AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_IMAGE_DEPLOYMENT,
    AZ_CONN_STR,
    AZ_CONTAINER,
    IMAGE_OUTPUT_DIR,
    IMAGE_TIMEOUT,
    DEFAULT_MAX_RETRIES,
    AZURE_IMAGE_SIZE,
    IMAGE_CONCURRENCY,
    IMAGE_MAX_RETRIES,
    IMAGE_RETRY_BACKOFF,
    IMAGE_BACKOFF_FACTOR,
    IMAGE_MAX_BACKOFF,
    LOGS_DIR,
)


@dataclass
class ImageTaskRecord:
    article_id: int
    topic: str
    status: str  # "SUCCESS" or "FAILURE"
    latency: float  # total seconds
    gen_duration: float = 0.0
    save_duration: float = 0.0
    blob_duration: float = 0.0
    blob_url: str = ""
    retries: int = 0
    error: str = ""
    timestamp: str = ""


class ImageMetricsTracker:
    """
    Thread-safe tracker for image generation latencies, breakdowns, transient 429 retries, and failures.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.records: list[ImageTaskRecord] = []
        self.total_429_retries: int = 0

    def reset(self):
        with self._lock:
            self.records.clear()
            self.total_429_retries = 0

    def record_retry(self, article_id: int, attempt: int, delay: float):
        with self._lock:
            self.total_429_retries += 1

    def record_success(
        self,
        article_id: int,
        topic: str,
        latency: float,
        gen_duration: float,
        save_duration: float,
        blob_duration: float,
        blob_url: str,
        retries: int = 0,
    ) -> ImageTaskRecord:
        record = ImageTaskRecord(
            article_id=article_id,
            topic=topic,
            status="SUCCESS",
            latency=round(latency, 2),
            gen_duration=round(gen_duration, 2),
            save_duration=round(save_duration, 2),
            blob_duration=round(blob_duration, 2),
            blob_url=blob_url,
            retries=retries,
            error="",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        with self._lock:
            self.records.append(record)
        return record

    def record_failure(
        self,
        article_id: int,
        topic: str,
        latency: float,
        error: str,
        retries: int = 0,
    ) -> ImageTaskRecord:
        record = ImageTaskRecord(
            article_id=article_id,
            topic=topic,
            status="FAILURE",
            latency=round(latency, 2),
            gen_duration=0.0,
            save_duration=0.0,
            blob_duration=0.0,
            blob_url="",
            retries=retries,
            error=str(error),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        with self._lock:
            self.records.append(record)
        return record

    def get_summary(self) -> dict[str, Any]:
        with self._lock:
            records = list(self.records)
            total_retries = self.total_429_retries

        total = len(records)
        succeeded = [r for r in records if r.status == "SUCCESS"]
        failed = [r for r in records if r.status == "FAILURE"]

        latencies = [r.latency for r in succeeded]
        gen_times = [r.gen_duration for r in succeeded if r.gen_duration > 0]
        save_times = [r.save_duration for r in succeeded if r.save_duration > 0]
        blob_times = [r.blob_duration for r in succeeded if r.blob_duration > 0]

        avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0
        min_latency = round(min(latencies), 2) if latencies else 0.0
        max_latency = round(max(latencies), 2) if latencies else 0.0
        avg_gen = round(sum(gen_times) / len(gen_times), 2) if gen_times else 0.0
        avg_save = round(sum(save_times) / len(save_times), 2) if save_times else 0.0
        avg_blob = round(sum(blob_times) / len(blob_times), 2) if blob_times else 0.0

        return {
            "total": total,
            "succeeded": len(succeeded),
            "failed": len(failed),
            "total_429_retries": total_retries,
            "avg_latency_sec": avg_latency,
            "min_latency_sec": min_latency,
            "max_latency_sec": max_latency,
            "avg_gen_sec": avg_gen,
            "avg_save_sec": avg_save,
            "avg_blob_sec": avg_blob,
            "failures": [
                {"article_id": r.article_id, "error": r.error, "latency": r.latency, "retries": r.retries}
                for r in failed
            ],
        }

    def print_summary(self):
        summary = self.get_summary()
        print("\n==================================================", flush=True)
        print("[IMAGE GENERATION METRICS SUMMARY]", flush=True)
        print(f"  Total Requested:       {summary['total']}", flush=True)
        print(f"  Succeeded:             {summary['succeeded']}", flush=True)
        print(f"  Failed:                {summary['failed']}", flush=True)
        print(f"  Transient 429 Retries: {summary['total_429_retries']}", flush=True)
        if summary['succeeded'] > 0:
            print(f"  Avg Latency:           {summary['avg_latency_sec']}s (min: {summary['min_latency_sec']}s, max: {summary['max_latency_sec']}s)", flush=True)
            print(f"  Breakdown (Avg):       AI Gen: {summary['avg_gen_sec']}s | Save: {summary['avg_save_sec']}s | Blob: {summary['avg_blob_sec']}s", flush=True)
        if summary['failed'] > 0:
            print(f"  Failures ({summary['failed']}):", flush=True)
            for f in summary['failures']:
                print(f"    - Article {f['article_id']}: {f['error'][:100]} (after {f['latency']}s, {f['retries']} retries)", flush=True)
        print("==================================================", flush=True)

    def export_csv(self, filepath: Path | str) -> str:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            records = list(self.records)

        fieldnames = [
            "timestamp",
            "article_id",
            "status",
            "latency_sec",
            "gen_sec",
            "save_sec",
            "blob_sec",
            "retries",
            "blob_url",
            "topic",
            "error",
        ]
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow({
                    "timestamp": r.timestamp,
                    "article_id": r.article_id,
                    "status": r.status,
                    "latency_sec": r.latency,
                    "gen_sec": r.gen_duration,
                    "save_sec": r.save_duration,
                    "blob_sec": r.blob_duration,
                    "retries": r.retries,
                    "blob_url": r.blob_url,
                    "topic": r.topic[:100],
                    "error": r.error.replace("\n", " ")[:500],
                })
        return str(filepath)

    def export_json(self, filepath: Path | str) -> str:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        summary = self.get_summary()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return str(filepath)


# Singleton metrics tracker instance
image_tracker = ImageMetricsTracker()


def _is_rate_limit_error(exc: Exception) -> bool:
    """Check if exception represents an HTTP 429 Too Many Requests error."""
    try:
        from openai import RateLimitError
        if isinstance(exc, RateLimitError):
            return True
    except ImportError:
        pass

    if getattr(exc, "status_code", None) == 429:
        return True

    resp = getattr(exc, "response", None)
    if resp is not None and getattr(resp, "status_code", None) == 429:
        return True

    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "too many requests" in msg


def _get_retry_delay(exc: Exception, attempt: int) -> float:
    """
    Extract Retry-After header or compute exponential backoff with full jitter.
    """
    resp = getattr(exc, "response", None)
    if resp is not None:
        headers = getattr(resp, "headers", None)
        if headers:
            retry_header = headers.get("retry-after") or headers.get("retry-after-ms")
            if retry_header:
                try:
                    val = float(retry_header)
                    if "ms" in str(retry_header).lower() or val > 1000:
                        val = val / 1000.0
                    return min(IMAGE_MAX_BACKOFF, val + random.uniform(0.5, 1.5))
                except (ValueError, TypeError):
                    pass

    # Exponential backoff: base * (factor ** attempt) + random jitter
    base_delay = IMAGE_RETRY_BACKOFF * (IMAGE_BACKOFF_FACTOR ** attempt)
    capped_delay = min(IMAGE_MAX_BACKOFF, base_delay)
    jitter = random.uniform(0.5, 2.0)
    return capped_delay + jitter


def gen_image_prompt(topic: str, summary: str) -> str:
    """Build editorial prompt for UPSC educational infographic."""
    return f"""
{topic}

{summary[:500]}

You're an AI capable of converting textual information into concrete images and your task is to convert the information in this article into symbolic images as you understand them.
Create a clean educational infographic for UPSC aspirants.

Style guidelines:
- Clear typography and concise educational labels are allowed
- Flat infographic or polished editorial infographic style
- Avoid cinematic photorealism
- Avoid fictional or highly detailed geographic maps
- Prefer schematic or symbolic map representations when needed
- Landscape orientation
- Suitable for serious civil services exam preparation
""".strip()


def get_image_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=CA_GEN_AZURE_OPENAI_API_KEY,
        api_version=CA_GEN_AZURE_OPENAI_API_VERSION,
        azure_endpoint=CA_GEN_AZURE_OPENAI_ENDPOINT,
        timeout=IMAGE_TIMEOUT,
        max_retries=0,  # Explicit application-level exponential backoff handles retries with telemetry
    )


def get_blob_service_client() -> BlobServiceClient:
    if not AZ_CONN_STR:
        raise ValueError("Azure Storage connection string is missing")
    return BlobServiceClient.from_connection_string(AZ_CONN_STR)


def generate_and_save_image(article_id: int, topic: str, summary: str) -> str:
    """
    Generate an infographic using Azure OpenAI (gpt-image-2.5-flare), save it locally,
    and upload it to Azure Blob Storage at thumbnail/{article_id}.jpg.
    Retries transient 429 errors with exponential backoff and records latency and failure metrics.
    Returns the uploaded blob URL.
    """
    t_start = time.time()
    retries_attempted = 0
    gen_duration = 0.0
    save_duration = 0.0
    blob_duration = 0.0

    try:
        print(f"[IMAGE] Starting background image task for article {article_id} ({AZURE_IMAGE_SIZE}): {topic[:50]}...", flush=True)

        image_client = get_image_client()
        prompt = gen_image_prompt(topic, summary)

        response = None
        for attempt in range(IMAGE_MAX_RETRIES + 1):
            try:
                t_gen = time.time()
                response = image_client.images.generate(
                    model=AZURE_OPENAI_IMAGE_DEPLOYMENT,
                    prompt=prompt,
                    size=AZURE_IMAGE_SIZE,
                )
                gen_duration = round(time.time() - t_gen, 2)
                break
            except Exception as e:
                if _is_rate_limit_error(e) and attempt < IMAGE_MAX_RETRIES:
                    retries_attempted += 1
                    delay = _get_retry_delay(e, attempt)
                    image_tracker.record_retry(article_id, attempt, delay)
                    print(
                        f"[IMAGE] [429 RATE LIMIT] Article {article_id}: Azure OpenAI rate limited (429). "
                        f"Retrying in {delay:.1f}s (retry {retries_attempted}/{IMAGE_MAX_RETRIES})...",
                        flush=True,
                    )
                    time.sleep(delay)
                else:
                    raise

        if response is None or not response.data:
            raise ValueError(f"Azure OpenAI returned empty response for article {article_id}")

        t_save = time.time()
        image_b64 = response.data[0].b64_json
        image_bytes = base64.b64decode(image_b64)

        image = PILImage.open(BytesIO(image_bytes)).convert("RGB")
        image_path = IMAGE_OUTPUT_DIR / f"{article_id}.jpg"

        image.save(
            str(image_path),
            format="JPEG",
            quality=82,
            optimize=True,
        )
        save_duration = round(time.time() - t_save, 2)

        blob_name = f"{article_id}.jpg"
        t_blob = time.time()
        blob_service = get_blob_service_client()
        blob_client = blob_service.get_blob_client(
            container=AZ_CONTAINER,
            blob=blob_name,
        )

        with open(str(image_path), "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        blob_url = blob_client.url
        blob_duration = round(time.time() - t_blob, 2)
        total_img_duration = round(time.time() - t_start, 2)

        retry_info = f" | Retries(429): {retries_attempted}" if retries_attempted > 0 else ""
        print(
            f"[IMAGE] Article {article_id} completed in {total_img_duration}s "
            f"[AI Gen: {gen_duration}s | Save: {save_duration}s | Blob: {blob_duration}s{retry_info}] -> {blob_url}",
            flush=True,
        )

        image_tracker.record_success(
            article_id=article_id,
            topic=topic,
            latency=total_img_duration,
            gen_duration=gen_duration,
            save_duration=save_duration,
            blob_duration=blob_duration,
            blob_url=blob_url,
            retries=retries_attempted,
        )
        return blob_url

    except Exception as e:
        total_fail_duration = round(time.time() - t_start, 2)
        print(f"[IMAGE] Image generation failed for article {article_id} after {total_fail_duration}s: {e}", flush=True)
        image_tracker.record_failure(
            article_id=article_id,
            topic=topic,
            latency=total_fail_duration,
            error=str(e),
            retries=retries_attempted,
        )
        raise e


async def async_generate_and_save_image(
    article_id: int,
    topic: str,
    summary: str,
    semaphore: asyncio.Semaphore | None = None,
) -> str:
    """
    Asynchronous wrapper for generate_and_save_image.
    Throttled by an asyncio.Semaphore to bound concurrency (default 3 concurrent requests).
    """
    if semaphore is not None:
        async with semaphore:
            return await asyncio.to_thread(generate_and_save_image, article_id, topic, summary)
    return await asyncio.to_thread(generate_and_save_image, article_id, topic, summary)


class ImageTaskQueue:
    """
    Asynchronous task queue for concurrent image generation.
    Maintains a worker pool with bounded concurrency (default: IMAGE_CONCURRENCY).
    """
    def __init__(self, concurrency: int = IMAGE_CONCURRENCY):
        self.concurrency = concurrency
        self.queue: asyncio.Queue = asyncio.Queue()
        self.workers: list[asyncio.Task] = []
        self._started = False

    async def start(self):
        if not self._started:
            self._started = True
            for i in range(self.concurrency):
                task = asyncio.create_task(self._worker(i), name=f"image_worker_{i}")
                self.workers.append(task)

    async def _worker(self, worker_id: int):
        while True:
            item = await self.queue.get()
            if item is None:
                self.queue.task_done()
                break
            article_id, topic, summary, fut = item
            try:
                res = await asyncio.to_thread(generate_and_save_image, article_id, topic, summary)
                if not fut.done():
                    fut.set_result(res)
            except Exception as ex:
                if not fut.done():
                    fut.set_exception(ex)
            finally:
                self.queue.task_done()

    async def submit(self, article_id: int, topic: str, summary: str) -> asyncio.Future:
        if not self._started:
            await self.start()
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        await self.queue.put((article_id, topic, summary, fut))
        return fut

    async def join(self):
        await self.queue.join()

    async def close(self):
        for _ in self.workers:
            await self.queue.put(None)
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        self._started = False
