import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="LLM Reliability Platform",
    page_icon="📊",
    layout="wide",
)


# =========================================================
# API CONFIGURATION
# =========================================================

API_BASE_URL = "https://llm-evaluation-platform-1.onrender.com"


# =========================================================
# PAGE TITLE
# =========================================================

st.title(
    "📊 Production AI Reliability Platform"
)

st.markdown("""
Enterprise-grade AI observability dashboard.

Features:
- LLM inference
- asynchronous evaluation
- drift detection
- reliability analytics
- statistical monitoring
""")


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header(
    "Platform Controls"
)

selected_model = st.sidebar.selectbox(
    "Select Model",
    [
        "llama-3.1-8b-instant",
    ],
)

refresh_button = st.sidebar.button(
    "Refresh Dashboard"
)


# =========================================================
# INFERENCE SECTION
# =========================================================

st.header("🚀 LLM Inference")

prompt = st.text_area(
    "Enter Prompt",
    height=150,
)

if st.button("Run Inference"):

    if not prompt.strip():

        st.warning(
            "Prompt cannot be empty."
        )

    else:

        payload = {
            "prompt": prompt,
            "model": selected_model,
        }

        try:

            with st.spinner(
                "Running inference..."
            ):

                response = requests.post(
                    f"{API_BASE_URL}/api/v1/inference",
                    json=payload,
                    timeout=60,
                )

            if response.status_code == 200:

                data = response.json()

                st.success(
                    "Inference completed."
                )

                # =====================================
                # RESPONSE
                # =====================================

                st.subheader(
                    "LLM Response"
                )

                st.write(
                    data["response"]
                )

                # =====================================
                # METADATA
                # =====================================

                st.subheader(
                    "Inference Metadata"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Request ID",
                        data["request_id"],
                    )

                with col2:

                    st.metric(
                        "Latency (s)",
                        data[
                            "latency_seconds"
                        ],
                    )

            else:

                st.error(
                    f"API Error: "
                    f"{response.text}"
                )

        except Exception as e:

            import traceback

            st.error(str(e))

            st.code(
                traceback.format_exc()
            )


# =========================================================
# DRIFT MONITORING
# =========================================================

st.header(
    "📈 Statistical Drift Monitoring"
)

try:

    drift_response = requests.get(
        f"{API_BASE_URL}/api/v1/drift",
        timeout=30,
    )

    if drift_response.status_code == 200:

        drift_data = (
            drift_response.json()
        )

        # =====================================
        # INSUFFICIENT DATA
        # =====================================

        if (
            drift_data["status"]
            == "insufficient_data"
        ):

            st.info(
                drift_data["message"]
            )

        else:

            # =====================================
            # METRICS
            # =====================================

            col1, col2, col3, col4 = (
                st.columns(4)
            )

            with col1:

                st.metric(
                    "Baseline Mean",
                    drift_data[
                        "baseline_mean"
                    ],
                )

            with col2:

                st.metric(
                    "Recent Mean",
                    drift_data[
                        "recent_mean"
                    ],
                )

            with col3:

                st.metric(
                    "KS Statistic",
                    drift_data[
                        "ks_statistic"
                    ],
                )

            with col4:

                st.metric(
                    "Wasserstein",
                    drift_data[
                        "wasserstein_distance"
                    ],
                )

            # =====================================
            # DRIFT STATUS
            # =====================================

            if drift_data[
                "drift_detected"
            ]:

                st.error(
                    "⚠️ Drift Detected"
                )

            else:

                st.success(
                    "✅ System Stable"
                )

            if drift_data[
                "severe_drift"
            ]:

                st.warning(
                    "🚨 Severe Drift Detected"
                )

            # =====================================
            # BAR CHART
            # =====================================

            st.subheader(
                "📊 Drift Indicators"
            )

            drift_metrics_df = (
                pd.DataFrame(
                    {
                        "Metric": [
                            "Baseline Mean",
                            "Recent Mean",
                            "KS Statistic",
                            "Wasserstein",
                        ],
                        "Value": [
                            drift_data[
                                "baseline_mean"
                            ],
                            drift_data[
                                "recent_mean"
                            ],
                            drift_data[
                                "ks_statistic"
                            ],
                            drift_data[
                                "wasserstein_distance"
                            ],
                        ],
                    }
                )
            )

            fig = px.bar(
                drift_metrics_df,
                x="Metric",
                y="Value",
                title=(
                    "AI Reliability Drift Metrics"
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            # =====================================
            # RELIABILITY GAUGE
            # =====================================

            reliability_score = (
                drift_data[
                    "recent_mean"
                ]
            )

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=reliability_score,
                    title={
                        "text": (
                            "AI Reliability Score"
                        )
                    },
                    gauge={
                        "axis": {
                            "range": [0, 10]
                        },
                    },
                )
            )

            st.plotly_chart(
                gauge,
                use_container_width=True,
            )

    else:

        st.error(
            "Failed to fetch drift analysis."
        )

except Exception as e:

    st.error(
        f"Drift endpoint error: {str(e)}"
    )


# =========================================================
# PLATFORM HEALTH
# =========================================================

st.header("🩺 Platform Health")

try:

    health_response = requests.get(
        f"{API_BASE_URL}/health",
        timeout=10,
    )

    if health_response.status_code == 200:

        health_data = (
            health_response.json()
        )

        st.success(
            f"Backend Status: "
            f"{health_data['status']}"
        )

        st.json(health_data)

    else:

        st.error(
            "Backend unhealthy."
        )

except Exception as e:

    st.error(
        f"Health check failed: {str(e)}"
    )


# =========================================================
# EVALUATION HISTORY
# =========================================================

st.header("📜 Evaluation History")

try:

    evaluation_response = requests.get(
        f"{API_BASE_URL}/api/v1/evaluations",
        timeout=30,
    )

    if evaluation_response.status_code == 200:

        evaluation_data = (
            evaluation_response.json()
        )

        evaluations = (
            evaluation_data["evaluations"]
        )

        # =====================================
        # EMPTY STATE
        # =====================================

        if len(evaluations) == 0:

            st.info(
                "No evaluations available yet."
            )

        else:

            st.success(
                f"Loaded "
                f"{len(evaluations)} evaluations"
            )

            # =====================================
            # TABLE ROWS
            # =====================================

            table_rows = []

            for item in evaluations:

                try:

                    row = {
                        "timestamp": item[
                            "timestamp"
                        ],

                        "prompt": item[
                            "prompt"
                        ][:50],

                        "overall_score":
                        item["evaluation"][
                            "overall_score"
                        ],

                        "correctness":
                        item["evaluation"][
                            "correctness"
                        ],

                        "clarity":
                        item["evaluation"][
                            "clarity"
                        ],

                        "helpfulness":
                        item["evaluation"][
                            "helpfulness"
                        ],

                        "safety":
                        item["evaluation"][
                            "safety"
                        ],

                        "professionalism":
                        item["evaluation"][
                            "professionalism"
                        ],
                    }

                    table_rows.append(row)

                except Exception as e:

                    st.error(str(e))

            # =====================================
            # DATAFRAME
            # =====================================

            history_df = pd.DataFrame(
                table_rows
            )

            st.dataframe(
                history_df,
                use_container_width=True,
            )

            # =====================================
            # TREND CHART
            # =====================================

            st.subheader(
                "📈 Reliability Trend"
            )

            trend_fig = px.line(
                history_df,
                y="overall_score",
                title=(
                    "Overall Reliability Trend"
                ),
            )

            st.plotly_chart(
                trend_fig,
                use_container_width=True,
            )

    else:

        st.error(
            "Failed to fetch evaluation history."
        )

except Exception as e:

    st.error(
        f"Evaluation history error: {str(e)}"
    )