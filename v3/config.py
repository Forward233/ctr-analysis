import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

BAYESIAN_OPT_PARAMS = {
    "n_estimators": (50, 500),
    "max_depth": (3, 30),
    "min_samples_split": (2, 20),
    "min_samples_leaf": (1, 10),
    "max_features": (0.1, 0.9),
}
BAYESIAN_OPT_ITER = 50
BAYESIAN_OPT_INIT = 10

LASSO_ALPHA_RANGE = [0.0001, 0.001, 0.01, 0.1]
SMOTE_SAMPLING_STRATEGY = 0.8

BASELINE_RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# ============================================================
# 快速调试配置（如需加速运行，取消下方注释并注释上方对应参数）
# ============================================================
# CV_FOLDS = 2
# BAYESIAN_OPT_PARAMS = {
#     "n_estimators": (50, 200),
#     "max_depth": (3, 15),
#     "min_samples_split": (2, 20),
#     "min_samples_leaf": (1, 10),
#     "max_features": (0.1, 0.9),
# }
# BAYESIAN_OPT_ITER = 5
# BAYESIAN_OPT_INIT = 3
# BASELINE_RF_PARAMS = {
#     "n_estimators": 50,
#     "max_depth": 10,
#     "min_samples_split": 5,
#     "min_samples_leaf": 2,
#     "random_state": RANDOM_STATE,
#     "n_jobs": -1,
# }
