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

        # Automatically get the expected features from the model
        try:
            model_features = getattr(model, 'feature_names_in_', None)
            if model_features is None:
                raise ValueError("Model does not expose feature names")
            st.sidebar.write(f"Expected features: {model_features}")
        except Exception as e:
            st.error(f"❌ Error extracting model features: {e}")
            model_features = []

        # Map model features to dataset columns (match by name similarity)
        matching_columns = []
        if model_features:
            for feature in model_features:
                matches = [col for col in data.columns if feature.lower() in col.lower()]
                if matches:
                    matching_columns.append((feature, matches[0]))  # Mapping the feature to the best match

        # If features do not match, ask user for manual mapping
        if not matching_columns and model_features:
            st.sidebar.warning("No matching columns found between the model and dataset. Please manually map them.")

        # Create a dictionary of column mappings if necessary
        column_mapping = {}
        if st.sidebar.button("Auto Map Columns") and matching_columns:
            for feature, match in matching_columns:
                column_mapping[feature] = match

        st.subheader("📋 Dataset Preview")
        with st.expander("🔎 View data sample"):
            st.dataframe(data.head())

        # If the user manually maps columns, apply the mapping
        if column_mapping:
            st.sidebar.write("Using the following column mappings:")
            st.sidebar.write(column_mapping)
            X = data[list(column_mapping.values())]
        else:
            # Fallback: Use model_features if available, else use all columns
            X = data[model_features] if model_features else data

        # Ensure dataset and model are compatible
        if X.shape[1] != len(model_features):
            st.error("Mismatch between model features and dataset columns.")
        else:
            y_pred = model(X)

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