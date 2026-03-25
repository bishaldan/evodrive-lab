from __future__ import annotations

import os
import time
from pathlib import Path

from app.storage.db import init_db
from app.storage.repository import next_queued_run
from app.worker.runner import execute_run


def main() -> None:
    init_db()
    poll_interval = float(os.getenv("EVODRIVE_POLL_INTERVAL", "5"))
    runs_dir = Path(os.getenv("EVODRIVE_RUNS_DIR", "runs"))
    reports_dir = Path(os.getenv("EVODRIVE_REPORTS_DIR", "reports"))
    runs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    while True:
        record = next_queued_run()
        if record is None:
            time.sleep(poll_interval)
            continue
        try:
            execute_run(record, runs_dir, reports_dir)
        except Exception:
            time.sleep(1)


if __name__ == "__main__":
    main()
