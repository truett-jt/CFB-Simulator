import pandas as pd
from pathlib import Path

# Pipeline & Metrics
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (log_loss)

# Models
from xgboost import XGBClassifier

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "modeling_data.csv"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

############ Data Loading ###############
df = pd.read_csv(DATA_PATH)
print(f"Orig DataFrame: {len(df):,} rows x {df.shape[1]} columns")

# drop those rows
df = df.dropna().copy()

print(f"Loaded data: {df.shape}")


############### Feature + Target Setup ###############
RANDOM_SEED = 42
TEST_SIZE = .2
VAL_SEASON = 2024

OUTCOME_CATEGORIES = ['PUNT', 'TD', 'TURNOVER', 'FG', 'DOWNS', 'MISSED_FG', 'END_HALF', 'END_GAME', 'TURNOVER_TD', 'OTHER', 'END_FOUR', 'SAFETY']
FEATURE_COLS = [
    # Drive context
    "startYardline",
    "startPeriod",
    'game_minutes_remaining',
    "startScoreDiff",
    "isHomeOffense",
    "neutralSite",

    # Team PPA values
    "ppa_off_overall",
    "ppa_off_passing",
    "ppa_off_rushing",
    "ppa_off_third_down",
    "ppa_def_overall",
    "ppa_def_passing",
    "ppa_def_rushing",
    "ppa_def_third_down",

    # Team strength
    "off_rating",
    "def_rating",
    "off_elo",
    "def_elo"
]

TARGET = 'drive_result'


############### Preprocessing ###############
X = df[FEATURE_COLS].copy()

# clean field position
if "startYardline" in X.columns:
    X['startYardline'] = X["startYardline"].clip(1, 99)

# encode target labels
le = LabelEncoder()
y = le.fit_transform(df[TARGET])

# Set up train/test split based on year
test_mask = df['season'] == VAL_SEASON
train_mask = df['season'] < VAL_SEASON

test_df = df[test_mask].copy()

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]
print(f"Training Data has {len(X_train)} rows")
print(f"Testing Data has {len(X_test)} rows")

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=RANDOM_SEED, stratify=y_train
)


############### Train Classifier ###############

xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=len(le.classes_),
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

param_dist = {
    "n_estimators": [200, 400, 600],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "min_child_weight": [1, 5, 10],
    "gamma": [0, 0.1, 0.3]
}

xgb_random = RandomizedSearchCV(
    xgb,
    param_distributions=param_dist,
    n_iter=25,
    scoring="neg_log_loss",
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

xgb_random.fit(X_tr, y_tr)


best_xgb = xgb_random.best_estimator_

print("Best Params:", xgb_random.best_params_)
print("Best CV Log-Loss:", -xgb_random.best_score_)

xgb_probs = best_xgb.predict_proba(X_val)
xgb_logloss = log_loss(y_val, xgb_probs)

print(f"XGBoost Validation Log-Loss: {xgb_logloss:.4f}")


############### Make predictions on 2024 Data and save to /reports ###############

test_probs = best_xgb.predict_proba(X_test)

probabilities_df = pd.DataFrame(
    test_probs,
    index=X_test.index,
    columns=le.classes_
)

probabilities_df["gameId"] = test_df.loc[probabilities_df.index, "gameId"]
probabilities_df["driveNumber"] = test_df.loc[probabilities_df.index, "driveNumber"]
probabilities_df = probabilities_df[["gameId", "driveNumber"] + list(le.classes_)]

output_path = REPORTS_DIR / "classification_preds.csv"
probabilities_df.to_csv(output_path, index=False)

print(f"Saved predictions to: {output_path}")



