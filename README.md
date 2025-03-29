# 📁 Project: ModelMirror
# Open-source Model Auditing & Fairness Dashboard

# 👤 Author: adeniyigiwa (https://github.com/adeniyigiwa)
# License: MIT

---

## 🔗 Live Demo
Try it instantly on Streamlit Cloud:  
👉 [https://modelmirror.streamlit.app](https://modelmirror.streamlit.app)

---

## 💡 What is ModelMirror?
**ModelMirror** is an open-source Streamlit dashboard for auditing and understanding your ML models. It helps data scientists, ML engineers, and responsible AI practitioners check their models for:

- ✅ Fairness across sensitive attributes (e.g., gender, race)
- 📊 Group-wise performance metrics (accuracy, selection rate)
- 🧠 Explainability using SHAP (global and per group)
- 🚨 Data leakage detection (correlation-based warnings)
- 📄 Exportable PDF audit reports

---

## ⚙️ How it Works
1. Upload a trained model (`.pkl`) and a dataset (`.csv`)
2. Select the target column and protected attribute (auto-detected)
3. Click **Run Audit** to:
   - View group fairness metrics
   - Analyze SHAP explanations
   - Check for potential leakage
   - Export a detailed report

---

## 📦 Features At-a-Glance
| Feature              | Description |
|----------------------|-------------|
| 🔍 Fairness Metrics  | Accuracy + selection rate by group using `fairlearn` |
| 🧠 SHAP Visuals       | Global + group-specific feature importance |
| ⚠️ Leakage Check     | Target leakage via correlation scan |
| 📄 PDF Reports       | One-click audit report export |
| 🔧 Model Support     | Scikit-learn, XGBoost, LightGBM |

---

## 📁 Folder Structure
# modelmirror/
# ├── app.py                  # Main Streamlit app
# ├── analyzer/
# │   ├── __init__.py
# │   ├── fairness.py         # Fairness metrics
# │   ├── explainability.py   # SHAP explainability
# │   └── leakage.py          # Leakage detection module
# ├── utils/
# │   ├── __init__.py
# │   ├── load_model.py       # Load models
# │   ├── pdf_export.py       # Export audit results to PDF
# │   └── test_fairness.py    # Unit test for fairness module
# ├── examples/
# │   ├── test_data.csv       # Placeholder
# │   └── test_model.pkl      # Placeholder
# ├── requirements.txt
# ├── README.md
# ├── CONTRIBUTING.md         # Community contribution guide
# ├── LICENSE
# └── .gitignore

---

## 🧪 Built With
- [Streamlit](https://streamlit.io/) — Interactive dashboard
- [Fairlearn](https://fairlearn.org/) — Fairness metrics
- [SHAP](https://shap.readthedocs.io/) — Explainability
- [scikit-learn](https://scikit-learn.org/) — ML baseline
- [XGBoost](https://xgboost.readthedocs.io/) / [LightGBM](https://lightgbm.readthedocs.io/) — Model support

---

## 🙌 Contributions Welcome
Want to add dark mode, more metrics, or polish the UI? PRs are welcome!

See [CONTRIBUTING.md](CONTRIBUTING.md) to get started.

---

## 📜 License
This project is open-sourced under the MIT License.