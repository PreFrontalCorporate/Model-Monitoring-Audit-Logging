import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from scipy.stats import ks_2samp, wasserstein_distance
import shap
import matplotlib.pyplot as plt
import sqlite3
from datetime import datetime
from tqdm import tqdm

# -----------------------------
# Data generation
# -----------------------------
def generate_data(n_samples=500, drift=False):
    X, y = make_classification(n_samples=n_samples, n_features=5, flip_y=0.01, random_state=None)
    if drift:
        X += np.random.normal(0.5, 0.3, X.shape)
    return X, y

# -----------------------------
# PSI calculation
# -----------------------------
def calculate_psi(expected, actual, buckets=10):
    expected_perc = np.histogram(expected, bins=buckets, range=(expected.min(), expected.max()), density=True)[0]
    actual_perc = np.histogram(actual, bins=buckets, range=(expected.min(), expected.max()), density=True)[0]
    expected_perc = np.where(expected_perc == 0, 0.0001, expected_perc)
    actual_perc = np.where(actual_perc == 0, 0.0001, actual_perc)
    psi_val = np.sum((expected_perc - actual_perc) * np.log(expected_perc / actual_perc))
    return np.abs(psi_val)

# -----------------------------
# Empirical KS threshold
# -----------------------------
def empirical_ks_threshold(ref_feature, alpha=0.05, num_permutations=500):
    stats = []
    n = len(ref_feature)
    for _ in tqdm(range(num_permutations), desc="Bootstrapping KS thresholds"):
        perm1 = np.random.choice(ref_feature, n, replace=True)
        perm2 = np.random.choice(ref_feature, n, replace=True)
        stat, _ = ks_2samp(perm1, perm2)
        stats.append(stat)
    return np.quantile(stats, 1 - alpha)

# -----------------------------
# Database setup
# -----------------------------
db_file = "audit_log.db"
conn = sqlite3.connect(db_file)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS audit_log (
    batch_num INTEGER,
    timestamp TEXT,
    accuracy REAL,
    auc REAL,
    logloss REAL,
    ks_stat REAL,
    wasserstein REAL,
    psi REAL,
    alert TEXT
)
""")
conn.commit()

# -----------------------------
# Train initial model
# -----------------------------
X_train, y_train = generate_data(1000)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
explainer = shap.TreeExplainer(model)

ref_X, _ = generate_data(500)
ks_threshold = empirical_ks_threshold(ref_X[:, 0], alpha=0.05, num_permutations=500)

# -----------------------------
# Monitoring loop
# -----------------------------
accuracy_list = []
ks_list = []
alert_batches = []
shap_distributions = []

for batch_num in range(1, 11):
    drift = batch_num >= 6
    X_batch, y_batch = generate_data(500, drift=drift)
    y_pred = model.predict(X_batch)
    y_proba = model.predict_proba(X_batch)[:, 1]

    acc = accuracy_score(y_batch, y_pred)
    auc = roc_auc_score(y_batch, y_proba)
    ll = log_loss(y_batch, y_proba)

    # Drift metrics
    ks_stats = []
    psi_vals = []
    w_dists = []
    p_vals = []

    for i in range(X_batch.shape[1]):
        ks_stat, p_value = ks_2samp(ref_X[:, i], X_batch[:, i])
        psi_val = calculate_psi(ref_X[:, i], X_batch[:, i])
        w_dist = wasserstein_distance(ref_X[:, i], X_batch[:, i])
        ks_stats.append(ks_stat)
        psi_vals.append(psi_val)
        w_dists.append(w_dist)
        p_vals.append(p_value)

    # Bonferroni correction
    corrected_alpha = 0.05 / X_batch.shape[1]
    reject = any([p < corrected_alpha for p in p_vals])
    max_ks = max(ks_stats)
    max_psi = max(psi_vals)
    max_w = max(w_dists)

    alert_flag = "DRIFT" if max_ks > ks_threshold or max_psi > 0.1 or max_w > 0.1 or reject else "OK"
    if alert_flag == "DRIFT":
        alert_batches.append(batch_num)

    cursor.execute("""
    INSERT INTO audit_log 
    (batch_num, timestamp, accuracy, auc, logloss, ks_stat, wasserstein, psi, alert)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (batch_num, datetime.now().isoformat(), acc, auc, ll, max_ks, max_w, max_psi, alert_flag))
    conn.commit()

    accuracy_list.append(acc)
    ks_list.append(max_ks)

    # SHAP feature importances
    shap_vals = explainer.shap_values(X_batch)
    shap_distributions.append(np.abs(shap_vals).mean(axis=1).mean(axis=0))

    print(f"Batch {batch_num}: Acc={acc:.3f}, AUC={auc:.3f}, LL={ll:.3f}, KS={max_ks:.3f}, W={max_w:.3f}, PSI={max_psi:.3f}, Alert={alert_flag}")

conn.close()

# -----------------------------
# Plot metrics
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(accuracy_list, label='Accuracy')
plt.plot(ks_list, label='Max KS Statistic')
plt.axhline(ks_threshold, color='red', linestyle='--', label='Empirical KS Threshold')
plt.title("Model Performance and Drift over Batches")
plt.xlabel("Batch Number")
plt.ylabel("Metric")
plt.legend()
plt.tight_layout()
plt.savefig("monitoring_plot.png")
plt.show()

# -----------------------------
# SHAP importance plot
# -----------------------------
avg_shap = np.mean(shap_distributions, axis=0)
plt.figure(figsize=(8, 4))
plt.bar(range(len(avg_shap)), avg_shap)
plt.title("Average SHAP Feature Importances across Batches")
plt.xlabel("Feature Index")
plt.ylabel("Mean Absolute SHAP Value")
plt.tight_layout()
plt.savefig("shap_importance.png")
plt.show()

print("✅ Monitoring complete! Audit logs in 'audit_log.db'. Plots and SHAP figures saved.")
