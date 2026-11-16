# FraudLens AI — Credit Card Fraud Intelligence (Streamlit + PySpark MLlib)

An academic, transparent credit card fraud analytics dashboard styled after business analytics Streamlit apps. Designed for Big Data Analytics viva demonstration.

## Features

- **Overview:** KPI cards, interactive class breakdown, amount histogram, activity graph.
- **Dataset Explorer:** filter and export previews, descriptive statistics, feature exploration.
- **Spark & BDA Lab:** actual Spark DataFrame aggregation and query plan when Spark is initialized, plus transparent conceptual partition illustration.
- **MLlib Training:** genuine PySpark Logistic Regression and Random Forest on uploaded ULB dataset, trained/tested with class-aware split. Shows confusion matrix, recall, precision, F1 and average precision.
- **Fraud Prediction:** held-out predictions, transaction cases, educational threshold exploration.
- **Stream Mining:** historical stream replay and functioning Bloom Filter, conceptual introductions to FM/DGIM.
- **Professor Demo:** guided viva script, project and architecture defenses.

## Data honesty

**The repository DOES NOT include the actual ULB data or a pretrained model.** In the absence of uploaded data, it shows an unmistakably labeled synthetic UI-preview dataset and disables training. Never present synthetic dashboard charts as measured fraud data. Upload the **real** ULB `creditcard.csv` through the sidebar to enable Spark MLlib training. No model performance values are fabricated.

Original dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Run locally

1. Install **Python 3.11 or 3.12** and **Java 17**.
2. Clone/extract this folder.
3. Create a virtual environment, then install packages.
4. Start Streamlit.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
java -version
streamlit run app.py
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
java -version
streamlit run app.py
```

For Windows Spark Java startup issues, ensure `JAVA_HOME` points to a valid JDK/JRE 17 installation and `java` is available on PATH. Try starting Java and Spark from the same terminal and inspect errors. This app uses Spark `local[2]`, so it does not need a Hadoop cluster.

## Deploy on Streamlit Community Cloud

1. Push all project files to a GitHub repository (do **not** upload the 150 MB CSV into source control by default).
2. Visit https://share.streamlit.io/ and click **Create app**.
3. Pick your repository, branch, and entry point `app.py`.
4. In advanced settings, choose **Python 3.11** if available, otherwise use a supported compatible version.
5. Deploy. The repository includes `requirements.txt` for PySpark and `packages.txt` requesting Java 17 on Linux.
6. The application will open in **synthetic preview** until you upload `creditcard.csv` in the sidebar.
7. Click **Spark & BDA Lab → Initialize Spark and analyze data**, then **MLlib Training → Train Spark MLlib model**.

**Important:** Free hosting is resource-constrained. Full dataset Spark MLlib training may exceed memory/time limits. Run locally before the viva, or reduce resource usage if needed. Streamlit may rerun or reset app state when asleep/restarted. For a guaranteed viva demo, keep a local backup ready. Cloud availability is not guaranteed by this repository alone.

## Deployment troubleshooting

- `JAVA_GATEWAY_EXITED`: Java unavailable or incompatible. Confirm `packages.txt` installed and `java -version` works. In cloud logs check the Java/PySpark startup error.
- Memory errors: use a machine with more RAM or compute bounded charts offline; the free instance may be insufficient for the full dataset.
- No predictions: upload real dataset, train a model first.
- Missing classes in a custom CSV: at least 10 rows of each class required for training; the original ULB dataset meets this requirement.
- Streamlit app initially showing different numbers than the ULB dataset: that is intentional **synthetic preview** mode.

## Important project distinctions

- **Spark local mode** is not a multi-node cluster.
- DataFrame transformations and actions use real Spark only after explicitly running the Spark lab or model training.
- **Stream replay** is historical event replay, **not** live bank transactions or Structured Streaming.
- **Bloom Filter** is implemented in the UI. FM and DGIM are documented, not implemented.
- PCA transformed V1–V28 features are provided by ULB; this program did not calculate those PCA components.
- PR-AUC metric displayed is scikit-learn **average precision (AP)** from actual held-out MLlib risk scores, not imaginary chart data.
- There is no tuning on a validation set in this prototype, and adjusting thresholds on test-set predictions is labeled educational only.

## Quick viva script

"This is FraudLens AI, a Spark MLlib fraud-detection and Big Data Analytics mini-project. We load the ULB transaction dataset into Spark DataFrames, examine class imbalance and data distributions, train a supervised classifier with an isolated test split, then evaluate fraud recall and precision. The app demonstrates Spark execution, and separately visualizes streaming concepts. A normal-only classifier could achieve 99.83% accuracy on the ULB dataset and still identify zero fraudulent payments."
