import csv
from datetime import datetime
from pathlib import Path


PREDICTION_LOG_PATH = Path("logs/predictions.csv")
PREDICTION_LOG_PATH.parent.mkdir(exist_ok=True)


def save_prediction_log(
    width: int,
    height: int,
    file_size_kb: float,
    recommended_action: str,
    recommended_format: str,
    recommended_quality: int,
    score: float | None,
    model_uri: str,
    run_id: str,
):
    is_new = not PREDICTION_LOG_PATH.exists()

    with open(PREDICTION_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if is_new:
            writer.writerow([
                "time",
                "width",
                "height",
                "file_size_kb",
                "recommended_action",
                "recommended_format",
                "recommended_quality",
                "score",
                "model_uri",
                "run_id",
            ])

        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            width,
            height,
            file_size_kb,
            recommended_action,
            recommended_format,
            recommended_quality,
            score,
            model_uri,
            run_id,
        ])