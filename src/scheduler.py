from __future__ import annotations

from datetime import datetime

import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from src.prediction_service import generate_latest_predictions


SCHEDULER_TIMEZONE = "Asia/Kolkata"
PREDICTION_HOUR = 18
PREDICTION_MINUTE = 0


def run_prediction_job() -> pd.DataFrame:
    """Run the daily production prediction job without retraining."""

    print(f"\n[{datetime.now().isoformat(timespec='seconds')}] Running prediction job...")

    predictions = generate_latest_predictions()

    print(predictions.to_string(index=False))
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Prediction job completed.")

    return predictions


def create_scheduler() -> BlockingScheduler:
    scheduler = BlockingScheduler(timezone=SCHEDULER_TIMEZONE)
    scheduler.add_job(
        run_prediction_job,
        CronTrigger(
            hour=PREDICTION_HOUR,
            minute=PREDICTION_MINUTE,
            timezone=SCHEDULER_TIMEZONE,
        ),
        id="daily_volatility_risk_prediction",
        name="Daily volatility risk prediction",
        replace_existing=True,
    )

    return scheduler


def main() -> None:
    scheduler = create_scheduler()

    print(
        "Scheduler started. "
        f"Daily predictions run at {PREDICTION_HOUR:02d}:{PREDICTION_MINUTE:02d} "
        f"{SCHEDULER_TIMEZONE}."
    )
    scheduler.start()


if __name__ == "__main__":
    main()
