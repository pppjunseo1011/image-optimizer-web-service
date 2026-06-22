from pathlib import Path

import pandas as pd
import streamlit as st


PREDICTION_LOG_PATH = Path("logs/predictions.csv")
FEEDBACK_LOG_PATH = Path("logs/feedback.csv")
LOW_CONFIDENCE_THRESHOLD = 0.65


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


st.set_page_config(
    page_title="Image Optimizer MLOps Dashboard",
    layout="wide",
)

st.title("Image Optimizer MLOps Monitoring Dashboard")

pred_df = read_csv_if_exists(PREDICTION_LOG_PATH)
feedback_df = read_csv_if_exists(FEEDBACK_LOG_PATH)

st.subheader("Prediction Monitoring")

col1, col2, col3, col4 = st.columns(4)

if not pred_df.empty:
    pred_df["score"] = pd.to_numeric(pred_df["score"], errors="coerce")

    total_requests = len(pred_df)
    avg_score = pred_df["score"].mean()
    low_confidence_count = len(pred_df[pred_df["score"] < LOW_CONFIDENCE_THRESHOLD])
    latest_model = pred_df["model_uri"].iloc[-1] if "model_uri" in pred_df.columns else "unknown"

    col1.metric("Total Requests", total_requests)
    col2.metric("Average Score", round(avg_score, 4))
    col3.metric("Low Confidence", low_confidence_count)
    col4.metric("Latest Model", latest_model)

    st.subheader("Score Trend")
    st.line_chart(pred_df["score"])

    st.subheader("Recommended Action Count")
    st.bar_chart(pred_df["recommended_action"].value_counts())

    st.subheader("Recommended Format Count")
    st.bar_chart(pred_df["recommended_format"].value_counts())

    st.subheader("Recent Predictions")
    st.dataframe(pred_df.tail(20), use_container_width=True)

else:
    col1.metric("Total Requests", 0)
    col2.metric("Average Score", "-")
    col3.metric("Low Confidence", 0)
    col4.metric("Latest Model", "-")
    st.info("아직 예측 로그가 없습니다. /recommend API를 먼저 실행하세요.")

st.divider()

st.subheader("User Feedback Monitoring")

col5, col6, col7 = st.columns(3)

if not feedback_df.empty:
    feedback_count = len(feedback_df)

    mismatch_df = feedback_df[
        (feedback_df["recommended_action"] != feedback_df["user_selected_action"])
        | (feedback_df["recommended_format"] != feedback_df["user_selected_format"])
    ]

    mismatch_count = len(mismatch_df)
    mismatch_rate = mismatch_count / feedback_count if feedback_count > 0 else 0

    col5.metric("Feedback Count", feedback_count)
    col6.metric("Mismatch Feedback", mismatch_count)
    col7.metric("Mismatch Rate", f"{mismatch_rate:.2%}")

    st.subheader("User Selected Action Count")
    st.bar_chart(feedback_df["user_selected_action"].value_counts())

    st.subheader("Recent Feedback")
    st.dataframe(feedback_df.tail(20), use_container_width=True)

    if not mismatch_df.empty:
        st.subheader("Mismatch Cases")
        st.dataframe(mismatch_df.tail(20), use_container_width=True)

else:
    col5.metric("Feedback Count", 0)
    col6.metric("Mismatch Feedback", 0)
    col7.metric("Mismatch Rate", "0.00%")
    st.info("아직 사용자 피드백 로그가 없습니다. /feedback API를 먼저 실행하세요.")