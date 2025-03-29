import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.load_model import load_any_model, load_any_dataset
from analyzer.fairness import evaluate_fairness
from analyzer.leakage import check_leakage
from utils.pdf_export import export_report
import shap

st.set_page_config(page_title="ModelMirror - Fairness Dashboard", layout="wide")
st.title("🔍 ModelMirror — Model Auditing & Fairness Dashboard")

with st.sidebar:
    st.header("📂 Upload Files")
    uploaded_model = st.file_uploader("Upload your trained model (.pkl, .py, .onnx)", type=["pkl", "py", "onnx"])
    uploaded_data = st.file_uploader("Upload your dataset (.csv, .xlsx, .parquet)", type=["csv", "xlsx", "parquet"])

if uploaded_model and uploaded_data:
    model = load_any_model(uploaded_model)
    data = load_any_dataset(uploaded_data)

    if model is not None and not data.empty:
        st.sidebar.success("✅ Data & model loaded")
        st.sidebar.write("Columns in data:", list(data.columns))

        common_targets = ["target", "income", "label"]
        common_protected = ["gender", "race", "sex", "ethnicity"]

        default_target = next((col for col in data.columns if col.lower() in common_targets), data.columns[-1])
        default_protected = next((col for col in data.columns if col.lower() in common_protected), data.columns[0])

        target_column = st.sidebar.selectbox("Select target column", data.columns, index=data.columns.get_loc(default_target))
        protected_column = st.sidebar.selectbox("Select protected attribute", data.columns, index=data.columns.get_loc(default_protected))

        st.subheader("📋 Dataset Preview")
        with st.expander("🔎 View data sample"):
            st.dataframe(data.head())

        if st.sidebar.button("Run Audit"):
            X = data.drop(columns=[target_column])
            y = data[target_column]
            y_pred = model.predict(X)

            fairness = evaluate_fairness(y, y_pred, data[protected_column])
            leakage_report = check_leakage(data, target_column)

            with st.expander("📊 Fairness Metrics"):
                st.json(fairness)
                metrics_df = pd.DataFrame(fairness["by_group"])
                for metric in metrics_df.columns:
                    fig, ax = plt.subplots()
                    sns.barplot(x=metrics_df.index, y=metrics_df[metric], ax=ax)
                    ax.set_title(f"{metric.capitalize()} by {protected_column}")
                    st.pyplot(fig)

            with st.expander("🧠 SHAP Explanation (Global)"):
                try:
                    explainer = shap.Explainer(model, X)
                    shap_values = explainer(X)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    shap.summary_plot(shap_values, X)
                    st.pyplot()
                except Exception as e:
                    st.warning(f"SHAP plot failed: {e}")

            with st.expander("🧠 SHAP by Group"):
                try:
                    unique_groups = data[protected_column].unique()
                    selected_group = st.selectbox("Select group", unique_groups)
                    group_data = X[data[protected_column] == selected_group]
                    if len(group_data) > 0:
                        group_explainer = shap.Explainer(model, group_data)
                        group_shap_values = group_explainer(group_data)
                        shap.summary_plot(group_shap_values, group_data)
                        st.pyplot()
                    else:
                        st.warning("No data points found for selected group.")
                except Exception as e:
                    st.error(f"SHAP group analysis failed: {e}")

            with st.expander("🚨 Potential Leakage Signals"):
                st.dataframe(leakage_report)

            with st.expander("📄 Export Audit Report"):
                if st.button("Download PDF Report"):
                    report_path = export_report(fairness, leakage_report)
                    with open(report_path, "rb") as file:
                        st.download_button(
                            label="📥 Download Report",
                            data=file,
                            file_name="ModelMirror_Audit_Report.pdf",
                            mime="application/pdf"
                        )