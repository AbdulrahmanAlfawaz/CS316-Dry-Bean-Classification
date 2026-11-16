"""CS316 dry bean classification. Run with python project.py."""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
SEED = 42

# Read the provided public data. The target contains seven bean varieties.
raw = pd.read_csv(ROOT / "data" / "dry_beans.csv")
missing = int(raw.isna().sum().sum())
if missing:
    raise ValueError("Unexpected missing data. Inspect the supplied dataset before training.")
# Exact duplicate records could otherwise appear in both training and test sets.
duplicates = int(raw.duplicated().sum())
df = raw.drop_duplicates().copy()
features = df.drop(columns="Class").columns.tolist()
# Downcast features and encode repeated labels to reduce dataframe memory.
before = int(df.memory_usage(deep=True).sum())
df[features] = df[features].astype("float32")
df["Class"] = df["Class"].astype("category")
after = int(df.memory_usage(deep=True).sum())
X = df[features]
y = df["Class"]
# Preserve class proportions. Reserve the test set before model selection.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED)
# Each validation fold learns its own scaler through the pipeline.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
candidates = {
    "Logistic regression": (LogisticRegression(max_iter=3000), {"model__C": [0.1, 1, 10]}),
    "K nearest neighbors": (KNeighborsClassifier(), {"model__n_neighbors": [3, 5, 9]}),
    "Decision tree": (DecisionTreeClassifier(random_state=SEED), {"model__max_depth": [5, 10, None]})
}
rows = []
searches = {}
for name, (model, parameters) in candidates.items():
    # Scaling is essential for distance-based models and harmless for a tree.
    pipeline = Pipeline([("scale", StandardScaler()), ("model", model)])
    search = GridSearchCV(pipeline, parameters, scoring="f1_macro", cv=cv, n_jobs=1)
    start = time.perf_counter()
    search.fit(X_train, y_train)
    index = search.best_index_
    rows.append({"model": name, "cv_macro_f1": float(search.best_score_),
                 "cv_std": float(search.cv_results_["std_test_score"][index]),
                 "parameters": str(search.best_params_), "fit_seconds": time.perf_counter()-start})
    searches[name] = search
# Choose the winner using training cross-validation only.
best_name = max(rows, key=lambda row: row["cv_macro_f1"])["model"]
model = searches[best_name].best_estimator_
predicted = model.predict(X_test)
probabilities = model.predict_proba(X_test)
# A majority-class baseline gives the task a meaningful reference point.
baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
base_pred = baseline.predict(X_test)
report = classification_report(y_test, predicted, output_dict=True)
pd.DataFrame(report).T.to_csv(OUT / "classification_report.csv")
pd.DataFrame(rows).to_csv(OUT / "model_comparison.csv", index=False)
# A fixed confidence rule demonstrates human review. It is not tuned on test data.
accepted = probabilities.max(axis=1) >= 0.80
review = {"threshold": 0.80, "coverage": float(accepted.mean()),
          "accepted_accuracy": float(accuracy_score(y_test[accepted], predicted[accepted])),
          "review_count": int((~accepted).sum())}
# Bootstrap the fixed test predictions for an approximate accuracy interval.
rng = np.random.default_rng(SEED)
correct = np.asarray(y_test == predicted)
boot = [correct[rng.integers(0, len(correct), len(correct))].mean() for _ in range(1000)]
summary = {"raw_rows": len(raw), "duplicates": duplicates, "clean_rows": len(df),
           "missing": missing, "features": len(features), "train_rows": len(X_train),
           "test_rows": len(X_test), "memory_before": before, "memory_after": after,
           "memory_reduction_pct": 100*(1-after/before), "best_model": best_name,
           "accuracy": accuracy_score(y_test, predicted), "macro_f1": f1_score(y_test, predicted, average="macro"),
           "baseline_accuracy": accuracy_score(y_test, base_pred),
           "accuracy_ci": np.quantile(boot, [0.025, 0.975]).tolist(), "review": review,
           "class_counts": {str(k): int(v) for k,v in y.value_counts().items()}}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
pd.DataFrame({"row_index": X_test.index, "actual": y_test, "predicted": predicted,
              "confidence": probabilities.max(axis=1)}).to_csv(OUT / "test_predictions.csv", index=False)
df.describe().to_csv(OUT / "data_summary.csv")
# Plot the training distribution and training-only feature relationships.
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
fig, ax = plt.subplots(figsize=(7,4))
y_train.value_counts().sort_index().plot.bar(ax=ax, color="#17736c")
ax.set(xlabel="Bean variety", ylabel="Training records", title="Training class distribution")
fig.tight_layout(); fig.savefig(OUT / "class_distribution.png"); plt.close(fig)
fig, ax = plt.subplots(figsize=(7,4))
for name in sorted(y_train.unique()):
    selected = y_train == name
    ax.scatter(X_train.loc[selected,"Area"], X_train.loc[selected,"Compactness"], s=4, alpha=0.4, label=name)
ax.set(xlabel="Area (pixels)", ylabel="Compactness", title="Training feature relationships")
ax.legend(fontsize=7, markerscale=3); fig.tight_layout(); fig.savefig(OUT / "feature_scatter.png"); plt.close(fig)
fig, ax = plt.subplots(figsize=(7,5))
ConfusionMatrixDisplay.from_predictions(y_test, predicted, ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
ax.set_title("Final model: held-out test confusion matrix")
fig.tight_layout(); fig.savefig(OUT / "confusion_matrix.png"); plt.close(fig)
fig, ax = plt.subplots(figsize=(7,4))
ax.barh([r["model"] for r in rows], [r["cv_macro_f1"] for r in rows], xerr=[r["cv_std"] for r in rows], color="#17736c")
ax.set(xlim=(0,1), xlabel="Five-fold CV macro F1", title="Training-only model selection")
fig.tight_layout(); fig.savefig(OUT / "model_comparison.png"); plt.close(fig)
print(json.dumps(summary, indent=2))
