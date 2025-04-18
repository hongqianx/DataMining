import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import optuna
import optuna.visualization
import xgboost as xgb

# Setup
logger = logging.getLogger("MLLogger")
logger.setLevel(logging.DEBUG)

# Mood label conversion
def split_mood_segments(mood):
    match mood:
        case mood if mood <= 4: return 0
        case mood if mood <= 6: return 1
        case mood if mood <= 8: return 2
        case mood if mood > 8: return 3
        case _:
            logger.error(f"Invalid mood value: {mood}")
            return -1

# Config
PREDICTION_COL = 'mood'
FEATURE_COLS = [
    'activity', 'appCat.builtin', 'appCat.communication', 'appCat.entertainment',
    'appCat.finance', 'appCat.game', 'appCat.office', 'appCat.other',
    'appCat.social', 'appCat.travel', 'appCat.unknown', 'appCat.utilities',
    'appCat.weather', 'call', 'circumplex.arousal', 'circumplex.valence',
    'screen', 'sms'
]
INPUT_DATA = "../input/df_rolling.csv"

# Load and prepare data
df = pd.read_csv(INPUT_DATA)
X = df[FEATURE_COLS].fillna(0)
y = df[PREDICTION_COL].apply(split_mood_segments)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tuning Strategy 1: GridSearchCV (RandomForest)
print("\n--- GridSearchCV ---")
rf_model = RandomForestClassifier(random_state=42)

grid_params = {
    'criterion': ['gini', 'entropy'],
    'n_estimators': [100, 150, 200],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 3],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt']
}

grid_search = GridSearchCV(rf_model, grid_params, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

y_pred_grid = grid_search.predict(X_test)
print("GridSearch Best Params:", grid_search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_grid))
print("F1 Score:", f1_score(y_test, y_pred_grid, average='weighted'))

# Tuning Strategy 2: Optuna (RandomForest)
print("\n--- Optuna Tuning (RandomForest) ---")
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

def rf_objective(trial):
    params = {
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "n_estimators": trial.suggest_int("n_estimators", 100, 200),
        "max_depth": trial.suggest_int("max_depth", 10, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 4),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 3),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "random_state": 42,
        "n_jobs": -1,
    }
    clf = RandomForestClassifier(**params)
    return cross_val_score(clf, X, y, cv=cv, scoring='accuracy').mean()

rf_study = optuna.create_study(direction="maximize")
rf_study.optimize(rf_objective, n_trials=30)

print("Optuna RF Best Params:", rf_study.best_params)
print("Best RF Score:", rf_study.best_value)
optuna.visualization.plot_optimization_history(rf_study).show()

# Tuning Strategy 3: Optuna with GPU XGBoost
print("\n--- Optuna Tuning (XGBoost GPU) ---")

def xgb_objective(trial):
    params = {
        "verbosity": 0,
        "objective": "multi:softmax",
        "num_class": len(set(y)),
        "tree_method": "gpu_hist",
        "n_estimators": trial.suggest_int("n_estimators", 100, 200),
        "max_depth": trial.suggest_int("max_depth", 6, 16),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
    }
    clf = xgb.XGBClassifier(**params)
    return cross_val_score(clf, X, y, cv=cv, scoring='accuracy').mean()

xgb_study = optuna.create_study(direction="maximize")
xgb_study.optimize(xgb_objective, n_trials=30)

print("Optuna XGBoost best params:", xgb_study.best_params)
print("Best XGBoost score:", xgb_study.best_value)
optuna.visualization.plot_optimization_history(xgb_study).show()

# Feature importance from GridSearch best model
importances = pd.Series(grid_search.best_estimator_.feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Feature Importances (RandomForest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()
