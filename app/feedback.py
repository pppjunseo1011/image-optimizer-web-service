import csv
from datetime import datetime
from pathlib import Path


FEEDBACK_LOG_PATH = Path("logs/feedback.csv")
FEEDBACK_LOG_PATH.parent.mkdir(exist_ok=True)


def save_feedback(
    recommended_action: str,
    user_selected_action: str,
    recommended_format: str,
    user_selected_format: str,
    score: float | None,
    comment: str = "",
):
    is_new = not FEEDBACK_LOG_PATH.exists()

    with open(FEEDBACK_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if is_new:
            writer.writerow([
                "time",
                "recommended_action",
                "user_selected_action",
                "recommended_format",
                "user_selected_format",
                "score",
                "comment",
            ])

        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            recommended_action,
            user_selected_action,
            recommended_format,
            user_selected_format,
            score,
            comment,
        ])