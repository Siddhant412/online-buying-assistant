from __future__ import annotations

from redis import Redis
from rq import Queue, Worker

from .settings import get_settings

DEFAULT_QUEUES = ("ingestion", "default")


def main() -> None:
    settings = get_settings()
    connection = Redis.from_url(settings.redis_url)
    queues = [Queue(name, connection=connection) for name in DEFAULT_QUEUES]
    worker = Worker(queues, connection=connection)
    worker.work()


if __name__ == "__main__":
    main()
