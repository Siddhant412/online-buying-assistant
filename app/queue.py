from __future__ import annotations

from redis import Redis
from rq import Queue

from .settings import get_settings


def enqueue_ingestion_job(job_id: str, queue_name: str = "ingestion") -> str:
    settings = get_settings()
    redis = Redis.from_url(settings.redis_url)
    queue = Queue(queue_name, connection=redis)
    queued_job = queue.enqueue(
        "app.ingestion_tasks.run_ingestion_job",
        job_id,
        job_timeout="30m",
        result_ttl=86400,
        failure_ttl=86400,
    )
    return queued_job.id
