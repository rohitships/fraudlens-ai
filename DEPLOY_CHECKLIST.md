# 5-minute deployment checklist

- [ ] GitHub repository contains `app.py`, `requirements.txt`, `packages.txt`, `.streamlit/config.toml`.
- [ ] Streamlit Cloud entry point is `app.py`.
- [ ] Set Python 3.11 if supported (otherwise 3.12).
- [ ] Initial UI shows **Synthetic preview mode** warning; that's expected.
- [ ] Upload real ULB `creditcard.csv` via sidebar and click **Load uploaded dataset**.
- [ ] Overview now shows exactly 284,807 total transactions and 492 frauds if the original full dataset was loaded.
- [ ] Click Spark Lab initialize and inspect actual Spark output.
- [ ] Click MLlib training, select Logistic Regression, train (may need local computer if cloud resources are constrained).
- [ ] Verify confusion matrix, actual held-out results and prediction page.
- [ ] Open Professor Demo page and rehearse 3-minute narrative.
- [ ] Keep an offline/local copy to avoid connectivity surprises.
