import json
import os
import subprocess
import sys

import streamlit as st


def run_prediction(text, use_pyvi, use_cuda):
    command = [
        sys.executable,
        "predict_worker.py",
        "--text",
        text,
    ]
    if use_pyvi:
        command.append("--use-pyvi")
    if use_cuda:
        command.append("--use-cuda")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=180,
    )

    stdout = completed.stdout.strip()
    if stdout:
        try:
            return json.loads(stdout.splitlines()[-1])
        except json.JSONDecodeError:
            pass

    error = completed.stderr.strip() or stdout or "Prediction process failed."
    raise RuntimeError(error)


st.title("Vietnamese Sentiment Analysis")

text = st.text_input("Input")
use_pyvi = st.checkbox("Use Vietnamese word segmentation", value=True)
use_cuda = st.checkbox("Use GPU (CUDA)", value=True)

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter a sentence.")
        st.stop()

    with st.spinner("Predicting..."):
        try:
            result = run_prediction(text, use_pyvi, use_cuda)
        except Exception as exc:
            st.error(str(exc))
            st.stop()

    st.caption(f"Running on: {result.get('device', 'unknown')}")
    st.success(result["label"])
    st.json({"probs": result["probs"]})
