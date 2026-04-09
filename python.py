"""
FINAL EEG MEDITATION CLASSIFIER - FULL UPGRADE
Improvements: LightGBM, targeted SMOTE, ratio features, feature selection, ensemble
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# -------------------------------
# CONFIG
# -------------------------------
RANDOM_SEED = 42
DATA_PATH = r"D:\FINAL_Improved_Meditation_Dataset.xls"

# -------------------------------
# 1. LOAD DATA
# -------------------------------
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
print("Class distribution:\n", df["Meditation_level"].value_counts())

feature_cols = [c for c in df.columns if c not in ["Meditation_level", "Subject"]]

# -------------------------------
# 2. SUBJECT-WISE NORMALIZATION
# -------------------------------
df[feature_cols] = df.groupby("Subject")[feature_cols].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-8)
)

# -------------------------------
# 3. RATIO FEATURE ENGINEERING (RESEARCH GOLD)
# -------------------------------
# Only create ratio if both columns exist in dataset
def safe_ratio(df, col_a, col_b, name):
    if col_a in df.columns and col_b in df.columns:
        df[name] = df[col_a] / (df[col_b] + 1e-5)
        print(f"  ✅ Created: {name}")
    else:
        print(f"  ⚠️  Skipped {name} (missing columns)")
    return df

print("\nCreating ratio features...")
df = safe_ratio(df, "Alpha_Frontal_Rel", "Beta_Frontal_Rel",  "Alpha_Beta_Ratio")
df = safe_ratio(df, "Theta_Frontal_Rel", "Alpha_Frontal_Rel", "Theta_Alpha_Ratio")
df = safe_ratio(df, "Delta_Frontal_Rel", "Gamma_Frontal_Rel", "Delta_Gamma_Ratio")
df = safe_ratio(df, "Theta_Frontal_Rel", "Beta_Frontal_Rel",  "Theta_Beta_Ratio")
df = safe_ratio(df, "Alpha_Parietal_Rel","Beta_Frontal_Rel",  "Parietal_Alpha_Beta_Ratio")

# Refresh feature cols after adding ratio features
feature_cols = [c for c in df.columns if c not in ["Meditation_level", "Subject"]]
print(f"\nTotal features after engineering: {len(feature_cols)}")

# -------------------------------
# 4. FEATURES & TARGET
# -------------------------------
le = LabelEncoder()
X = df[feature_cols].copy()
y = le.fit_transform(df["Meditation_level"])  # NumPy array

# Fix label issue: ensure [0, 1, 2]
if y.min() > 0:
    y = y - y.min()

print("\nUnique classes:", np.unique(y))
print("Class counts:  ", np.bincount(y))

# Replace inf/nan
X = X.replace([np.inf, -np.inf], np.nan)

groups = df["Subject"]

# -------------------------------
# 5. TRAIN-TEST SPLIT (SUBJECT-AWARE)
# -------------------------------
gss = GroupShuffleSplit(test_size=0.2, random_state=RANDOM_SEED)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y[train_idx], y[test_idx]   # ✅ NumPy indexing

print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")
print("Train class dist:", np.bincount(y_train))
print("Test  class dist:", np.bincount(y_test))

# -------------------------------
# 6. SMART SMOTE STRATEGY
#    Oversample minority class more aggressively
# -------------------------------
counts = np.bincount(y_train)
majority = int(counts.max())

# Each class gets boosted to at least 80% of majority
sampling_strategy = {
    i: max(counts[i], int(majority * 0.80))
    for i in range(len(counts))
    if counts[i] < majority
}
# With:
sampling_strategy = {2: 80}  # ✅ Force class 2 to have 80 samples (adjust as needed)
print("\nSMOTE sampling strategy:", sampling_strategy)

# Determine safe k_neighbors (must be < smallest class count)
min_class_count = int(counts.min())
k_neighbors = max(1, min(5, min_class_count - 1))
print(f"SMOTE k_neighbors: {k_neighbors}")

smote = SMOTE(
    sampling_strategy=sampling_strategy,
    k_neighbors=k_neighbors,
    random_state=RANDOM_SEED
)

# -------------------------------
# 7. DEFINE THREE STRONG MODELS
# -------------------------------
lgbm = LGBMClassifier(
    n_estimators=800,
    learning_rate=0.02,
    max_depth=6,
    num_leaves=31,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.3,
    reg_lambda=0.5,
    class_weight="balanced",
    random_state=RANDOM_SEED,
    verbose=-1
)

xgb = XGBClassifier(
    n_estimators=600,
    learning_rate=0.02,
    max_depth=5,
    subsample=0.85,
    colsample_bytree=0.85,
    gamma=0.3,
    reg_alpha=0.3,
    reg_lambda=1.0,
    eval_metric="mlogloss",
    n_jobs=-1,
    random_state=RANDOM_SEED
)

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=8,
    class_weight="balanced",
    n_jobs=-1,
    random_state=RANDOM_SEED
)

# Soft voting ensemble — combines all 3 model probabilities
ensemble = VotingClassifier(
    estimators=[
        ("lgbm", lgbm),
        ("xgb",  xgb),
        ("rf",   rf)
    ],
    voting="soft"
)

# -------------------------------
# 8. PIPELINE
# -------------------------------
pipeline = Pipeline([
    ("imputer",   SimpleImputer(strategy="median")),
    ("scaler",    StandardScaler()),
    ("select",    SelectKBest(f_classif, k=15)),    # ✅ Keep top 15 features
    ("smote",     smote),
    ("model",     ensemble)                         # ✅ Ensemble model
])

# -------------------------------
# 9. CROSS VALIDATION
# -------------------------------
print("\n--- 5-Fold Stratified Cross Validation ---")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
cv_bal_acc, cv_f1 = [], []

for fold, (tr_i, val_i) in enumerate(cv.split(X_train, y_train), 1):

    X_tr,  X_val = X_train.iloc[tr_i],  X_train.iloc[val_i]
    y_tr,  y_val = y_train[tr_i],        y_train[val_i]

    pipeline.fit(X_tr, y_tr)
    preds = pipeline.predict(X_val)

    bal = balanced_accuracy_score(y_val, preds)
    f1  = f1_score(y_val, preds, average="macro", zero_division=0)

    cv_bal_acc.append(bal)
    cv_f1.append(f1)

    # Per-class F1 for monitoring
    per_class = f1_score(y_val, preds, average=None, zero_division=0)
    print(f"  Fold {fold} → Bal Acc: {bal:.4f} | Macro F1: {f1:.4f} | Per-class F1: {np.round(per_class, 3)}")

print(f"\nMean CV Balanced Accuracy : {np.mean(cv_bal_acc):.4f} ± {np.std(cv_bal_acc):.4f}")
print(f"Mean CV Macro F1          : {np.mean(cv_f1):.4f} ± {np.std(cv_f1):.4f}")

# -------------------------------
# 10. FINAL TRAINING
# -------------------------------
print("\n--- Training Final Model on Full Train Set ---")
pipeline.fit(X_train, y_train)

# -------------------------------
# 11. TEST EVALUATION
# -------------------------------
print("\n--- Final Test Set Results ---")
preds = pipeline.predict(X_test)

bal_acc = balanced_accuracy_score(y_test, preds)
f1      = f1_score(y_test, preds, average="macro", zero_division=0)
per_cls = f1_score(y_test, preds, average=None,    zero_division=0)

print(f"\n{'='*40}")
print(f"  Balanced Accuracy  : {round(bal_acc, 4)}")
print(f"  Macro F1 Score     : {round(f1, 4)}")
print(f"  Per-class F1       : {np.round(per_cls, 4)}")
print(f"{'='*40}")

print("\nDetailed Classification Report:")
print(classification_report(
    y_test, preds,
    target_names=[str(c) for c in le.classes_],
    zero_division=0
))

# -------------------------------
# 12. FEATURE IMPORTANCE (for paper)
# -------------------------------
print("\n--- Feature Importance (via LGBM) ---")
try:
    selector  = pipeline.named_steps["select"]
    ensemble_ = pipeline.named_steps["model"]
    lgbm_     = ensemble_.estimators_[0]           # LGBM is first estimator

    sel_mask  = selector.get_support()
    sel_feats = np.array(feature_cols)[sel_mask]

    feat_imp  = pd.DataFrame({
        "Feature":    sel_feats,
        "Importance": lgbm_.feature_importances_
    }).sort_values("Importance", ascending=False)

    print("\nTop 15 Most Important Features (LGBM):")
    print(feat_imp.head(15).to_string(index=False))

    # Check if your ratio features made the cut
    ratio_feats = [f for f in feat_imp["Feature"] if "Ratio" in f]
    print(f"\nRatio features in top selection: {ratio_feats if ratio_feats else 'None (may not exist in your dataset)'}")

except Exception as e:
    print(f"[Feature importance skipped: {e}]")