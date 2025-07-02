# 🛡️ Model Monitoring & Audit Logging

A robust, theoretically rigorous Python module for continuous model performance monitoring and data drift detection, designed for financial ML models and other high-stakes environments.

## 💡 Motivation

In regulated settings (e.g., finance, healthcare), it is critical to continuously monitor machine learning models for:

* Accuracy decay and performance degradation
* Data distribution shifts (data drift)
* Changes in relationships between features and targets (concept drift)

This module implements advanced statistical tests, audit logging, and interpretability methods to ensure models remain valid, fair, and explainable over time.

---

## ⚙️ Features

* **Empirical Kolmogorov-Smirnov (KS) thresholds** estimated via permutation bootstrapping.
* **Population Stability Index (PSI)** and **Wasserstein distance** to detect subtle distribution changes.
* **Bonferroni correction** for multi-feature hypothesis testing, ensuring statistical rigor.
* **SHAP feature importance drift analysis**, providing interpretability on which features contribute most to detected drifts.
* **Bootstrap-ready confidence intervals** (extensible), supporting uncertainty quantification.
* **Automated logging** to an SQLite audit database for reproducibility and regulatory traceability.
* **Visual plots** for performance metrics and feature importance over time.

---

## 🧬 Theoretical foundation

The module implements:

* Formal hypothesis tests on feature distributions:

  $$
  H_0: F_0 = F_1 \quad \text{vs.} \quad H_A: F_0 \neq F_1
  $$
* Empirical threshold estimation for KS to avoid arbitrary cutoffs.
* Family-wise error control using Bonferroni correction.
* Proven convergence of PSI to KL-divergence under fine binning.
* SHAP-based feature contribution analysis to monitor internal model logic.

---

## 🚀 How it works

1. Train a baseline model on initial data (`D0`).
2. For each incoming batch:

   * Compute performance metrics (Accuracy, AUC, Log Loss).
   * Test each feature for drift using KS, PSI, and Wasserstein.
   * Apply Bonferroni-corrected p-values.
   * Flag batch as `DRIFT` if any drift metric exceeds thresholds.
   * Log metrics and predictions to an SQLite audit database.
   * Compute SHAP values for interpretability.
3. Update drift plots and SHAP feature importance charts.

---

## 📄 Outputs

* **`audit_log.db`**: Full batch-wise metrics and drift flags.
* **`monitoring_plot.png`**: Accuracy and drift statistic trends.
* **`shap_importance.png`**: Average SHAP feature importances across batches.

---

## 🧑‍💻 Running the script

```bash
python model-monitoring-audit-logging.py
```

Dependencies:

* numpy
* pandas
* scikit-learn
* scipy
* matplotlib
* shap
* tqdm
* sqlite3 (standard library)

---

## 📈 Example outputs

```
Batch 1: Acc=0.660, AUC=0.721, LL=0.640, KS=0.456, W=1.032, PSI=11.715, Alert=DRIFT
Batch 2: Acc=0.606, AUC=0.471, LL=0.954, KS=0.486, W=0.947, PSI=13.887, Alert=DRIFT
...
✅ Monitoring complete! Audit logs in 'audit_log.db'. Plots and SHAP figures saved.
```

---

## ✅ Why it works

* **Empirical KS threshold** ensures robust detection under finite samples.
* **Bonferroni correction** rigorously controls error rate when testing multiple features.
* **PSI and Wasserstein** capture nuanced changes not visible via single metrics.
* **Bootstrap capability** provides rigorous confidence intervals (extensible).
* **SHAP interpretability** reveals model logic shifts and feature attribution changes.

---

## 💬 Questions or contributions

Open issues or pull requests — contributions to improve metrics, add new drift detection methods, or integrate live dashboards are welcome!

---

## 🛡️ License

MIT License. Free to use and modify for research, audits, or production.

---
