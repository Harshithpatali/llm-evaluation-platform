import pandas as pd
import plotly.express as px
import requests
import streamlit as st


# =========================================================
# CONFIGURATION
# =========================================================

BACKEND_URL = (
    "http://127.0.0.1:8000"
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title=(
        "AI Reliability Platform"
    ),
    layout="wide",
)


# =========================================================
# TITLE
# =========================================================

st.title(
    "Production AI Reliability Platform"
)

st.markdown(
    """
    Real-time AI observability,
    evaluation, and drift monitoring.
    """
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header(
    "System Status"
)

st.sidebar.success(
    "Frontend Online"
)


# =========================================================
# INFERENCE SECTION
# =========================================================

st.header(
    "LLM Inference"
)

prompt = st.text_area(
    "Enter Prompt",
    height=150
)

if st.button("Run Inference"):

    if prompt.strip():

        with st.spinner(
            "Generating response..."
        ):

            try:

                response = requests.post(
                    f"{BACKEND_URL}"
                    "/api/v1/inference",
                    json={
                        "prompt": prompt
                    },
                    timeout=60,
                )

                result = response.json()

                st.subheader(
                    "Model Response"
                )

                st.write(
                    result["response"]
                )

            except Exception as e:

                st.error(str(e))


# =========================================================
# EVALUATION HISTORY
# =========================================================

st.header(
    "Evaluation History"
)

try:

    response = requests.get(
        f"{BACKEND_URL}"
        "/api/v1/evaluations"
    )

    data = response.json()

    evaluations = data.get(
        "evaluations",
        []
    )

    if evaluations:

        df = pd.DataFrame(
            evaluations
        )

        st.dataframe(
            df,
            use_container_width=True
        )

        # =============================================
        # SCORE TREND CHART
        # =============================================

        if "overall_score" in df.columns:

            st.subheader(
                "Overall Score Trend"
            )

            df["index"] = range(len(df))

            fig = px.line(
                df,
                x="index",
                y="overall_score",
                title=(
                    "LLM Quality Trend"
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

except Exception as e:

    st.error(
        f"Evaluation fetch failed: "
        f"{e}"
    )


# =========================================================
# DRIFT MONITORING
# =========================================================

st.header(
    "Statistical Drift Detection"
)

try:

    response = requests.get(
        f"{BACKEND_URL}"
        "/api/v1/drift"
    )

    drift_data = response.json()

    if (
        "drift_detected"
        in drift_data
    ):

        drift_detected = (
            drift_data[
                "drift_detected"
            ]
        )

        if drift_detected:

            st.error(
                "Drift Detected"
            )

        else:

            st.success(
                "No Drift Detected"
            )

        # =============================================
        # METRICS
        # =============================================

        col1, col2, col3 = (
            st.columns(3)
        )

        col1.metric(
            "KS Statistic",
            round(
                drift_data[
                    "ks_statistic"
                ],
                4
            )
        )

        col2.metric(
            "P-Value",
            round(
                drift_data[
                    "p_value"
                ],
                4
            )
        )

        col3.metric(
            "Wasserstein Distance",
            round(
                drift_data[
                    "wasserstein_distance"
                ],
                4
            )
        )

        # =============================================
        # MEAN COMPARISON
        # =============================================

        st.subheader(
            "Distribution Comparison"
        )

        comparison_df = pd.DataFrame({

            "Window": [
                "Baseline",
                "Current"
            ],

            "Mean Score": [

                drift_data[
                    "baseline_mean"
                ],

                drift_data[
                    "current_mean"
                ],
            ],
        })

        fig = px.bar(
            comparison_df,
            x="Window",
            y="Mean Score",
            title=(
                "Baseline vs Current"
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "Insufficient data "
            "for drift analysis."
        )

except Exception as e:

    st.error(
        f"Drift analysis failed: {e}"
    )