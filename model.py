import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    roc_auc_score, log_loss, precision_score, recall_score,
    f1_score, accuracy_score, roc_curve, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from bayes_opt import BayesianOptimization
import xgboost as xgb
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

from config import (
    RANDOM_STATE, CV_FOLDS, BAYESIAN_OPT_PARAMS, BAYESIAN_OPT_ITER,
    BAYESIAN_OPT_INIT, SMOTE_SAMPLING_STRATEGY, BASELINE_RF_PARAMS,
    FIGURE_DIR, OUTPUT_DIR
)

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def apply_smote(X_train, y_train):
    smote = SMOTE(sampling_strategy=SMOTE_SAMPLING_STRATEGY, random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"[SMOTE] 原始样本: {len(y_train)} (正样本: {sum(y_train)})")
    print(f"[SMOTE] 重采样后: {len(y_res)} (正样本: {sum(y_res)})")
    return X_res, y_res


def train_baseline_rf(X_train, y_train):
    rf = RandomForestClassifier(**BASELINE_RF_PARAMS)
    rf.fit(X_train, y_train)
    return rf


def bayesian_optimize_rf(X_train, y_train):
    def rf_evaluate(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features):
        params = {
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth),
            "min_samples_split": int(min_samples_split),
            "min_samples_leaf": int(min_samples_leaf),
            "max_features": max_features,
            "random_state": RANDOM_STATE,
            "n_jobs": 8,
        }
        rf = RandomForestClassifier(**params)
        scores = cross_val_score(rf, X_train, y_train, cv=CV_FOLDS, scoring="roc_auc", n_jobs=8)
        return scores.mean()

    optimizer = BayesianOptimization(
        f=rf_evaluate,
        pbounds=BAYESIAN_OPT_PARAMS,
        random_state=RANDOM_STATE,
        verbose=2,
    )
    optimizer.maximize(init_points=BAYESIAN_OPT_INIT, n_iter=BAYESIAN_OPT_ITER)

    best_params = optimizer.max["params"]
    best_params["n_estimators"] = int(best_params["n_estimators"])
    best_params["max_depth"] = int(best_params["max_depth"])
    best_params["min_samples_split"] = int(best_params["min_samples_split"])
    best_params["min_samples_leaf"] = int(best_params["min_samples_leaf"])
    best_params["random_state"] = RANDOM_STATE
    best_params["n_jobs"] = -1

    print(f"[贝叶斯优化] 最优参数: {best_params}")
    print(f"[贝叶斯优化] 最优AUC: {optimizer.max['target']:.4f}")

    optimized_rf = RandomForestClassifier(**best_params)
    optimized_rf.fit(X_train, y_train)

    return optimized_rf, best_params, optimizer


def apply_ccp_pruning(X_train, y_train, X_test, y_test, best_params):
    pruning_params = {k: v for k, v in best_params.items() if k != "n_jobs"}
    pruning_params["n_jobs"] = 1

    dt = DecisionTreeClassifier(
        max_depth=pruning_params.get("max_depth", 10),
        min_samples_split=pruning_params.get("min_samples_split", 5),
        min_samples_leaf=pruning_params.get("min_samples_leaf", 2),
        random_state=RANDOM_STATE,
    )
    dt.fit(X_train, y_train)

    path = dt.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas
    impurities = path.impurities

    best_alpha = 0
    best_auc = 0
    alpha_scores = []

    sample_alphas = np.linspace(0, ccp_alphas.max() * 0.5, min(20, len(ccp_alphas)))
    for alpha in sample_alphas:
        rf_pruned = RandomForestClassifier(
            n_estimators=best_params.get("n_estimators", 100),
            max_depth=best_params.get("max_depth", 10),
            min_samples_split=best_params.get("min_samples_split", 5),
            min_samples_leaf=best_params.get("min_samples_leaf", 2),
            max_features=best_params.get("max_features", 0.5),
            ccp_alpha=alpha,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        rf_pruned.fit(X_train, y_train)
        y_prob = rf_pruned.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        alpha_scores.append((alpha, auc))
        if auc > best_auc:
            best_auc = auc
            best_alpha = alpha

    print(f"[CCP剪枝] 最优ccp_alpha: {best_alpha:.6f}, AUC: {best_auc:.4f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    alphas_plot, aucs_plot = zip(*alpha_scores)
    ax.plot(alphas_plot, aucs_plot, marker="o", color="green")
    ax.axvline(x=best_alpha, color="red", linestyle="--", label=f"最优alpha={best_alpha:.6f}")
    ax.set_xlabel("CCP Alpha")
    ax.set_ylabel("AUC")
    ax.set_title("CCP剪枝参数选择")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "ccp_pruning.png"), dpi=150, bbox_inches="tight")
    plt.close()

    final_rf = RandomForestClassifier(
        n_estimators=best_params.get("n_estimators", 100),
        max_depth=best_params.get("max_depth", 10),
        min_samples_split=best_params.get("min_samples_split", 5),
        min_samples_leaf=best_params.get("min_samples_leaf", 2),
        max_features=best_params.get("max_features", 0.5),
        ccp_alpha=best_alpha,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    final_rf.fit(X_train, y_train)

    return final_rf, best_alpha


def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": model_name,
        "AUC": roc_auc_score(y_test, y_prob),
        "LogLoss": log_loss(y_test, y_prob),
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
    }
    return metrics, y_pred, y_prob


def train_comparison_models(X_train, y_train, X_test, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
        "GBDT": GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=RANDOM_STATE
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=RANDOM_STATE, verbose=-1
        ),
    }

    results = []
    model_probs = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)
        model_probs[name] = y_prob
        print(f"[{name}] AUC={metrics['AUC']:.4f}, LogLoss={metrics['LogLoss']:.4f}, F1={metrics['F1-Score']:.4f}")

    return results, model_probs


def plot_model_comparison(all_results):
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics_to_plot = ["AUC", "LogLoss", "Accuracy", "Precision", "Recall", "F1-Score"]
    colors = plt.cm.Set2(np.linspace(0, 1, len(df_results)))

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 3][idx % 3]
        bars = ax.bar(df_results["Model"], df_results[metric], color=colors)
        ax.set_title(metric, fontsize=13)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        for bar, val in zip(bars, df_results[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("模型性能对比", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curves(y_test, model_probs, optimized_rf_prob, final_rf_prob):
    fig, ax = plt.subplots(figsize=(10, 8))

    fpr, tpr, _ = roc_curve(y_test, final_rf_prob)
    auc_val = roc_auc_score(y_test, final_rf_prob)
    ax.plot(fpr, tpr, linewidth=2.5, label=f"优化随机森林 (AUC={auc_val:.4f})")

    fpr2, tpr2, _ = roc_curve(y_test, optimized_rf_prob)
    auc_val2 = roc_auc_score(y_test, optimized_rf_prob)
    ax.plot(fpr2, tpr2, linewidth=1.5, linestyle="--", label=f"贝叶斯优化RF (AUC={auc_val2:.4f})")

    for name, probs in model_probs.items():
        fpr_m, tpr_m, _ = roc_curve(y_test, probs)
        auc_m = roc_auc_score(y_test, probs)
        ax.plot(fpr_m, tpr_m, linewidth=1, alpha=0.7, label=f"{name} (AUC={auc_m:.4f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="随机分类器")
    ax.set_xlabel("假正例率 (FPR)", fontsize=12)
    ax.set_ylabel("真正例率 (TPR)", fontsize=12)
    ax.set_title("ROC曲线对比", fontsize=14)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "roc_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(y_test, y_pred, model_name="优化随机森林"):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["未点击", "点击"], yticklabels=["未点击", "点击"])
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    ax.set_title(f"{model_name} - 混淆矩阵")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()


def run_ablation_study(X_train, y_train, X_test, y_test, best_params,
                       selected_features, X_train_full, ccp_alpha):
    results = []

    rf_base = RandomForestClassifier(**BASELINE_RF_PARAMS)
    rf_base.fit(X_train_full, y_train)
    m, _, _ = evaluate_model(rf_base, X_test, y_test, "基线RF")
    results.append(m)

    rf_lasso = RandomForestClassifier(**BASELINE_RF_PARAMS)
    rf_lasso.fit(X_train, y_train)
    m, _, _ = evaluate_model(rf_lasso, X_test[X_train.columns], y_test, "+LASSO特征筛选")
    results.append(m)

    X_sm, y_sm = apply_smote(X_train, y_train)
    rf_smote = RandomForestClassifier(**BASELINE_RF_PARAMS)
    rf_smote.fit(X_sm, y_sm)
    m, _, _ = evaluate_model(rf_smote, X_test[X_train.columns], y_test, "+SMOTE重采样")
    results.append(m)

    best_p = {k: v for k, v in best_params.items()}
    best_p["n_jobs"] = -1
    best_p["random_state"] = RANDOM_STATE
    rf_bayes = RandomForestClassifier(**best_p)
    rf_bayes.fit(X_sm, y_sm)
    m, _, _ = evaluate_model(rf_bayes, X_test[X_train.columns], y_test, "+贝叶斯优化")
    results.append(m)

    best_p_ccp = {k: v for k, v in best_p.items()}
    best_p_ccp["ccp_alpha"] = ccp_alpha
    rf_all = RandomForestClassifier(**best_p_ccp)
    rf_all.fit(X_sm, y_sm)
    m, _, _ = evaluate_model(rf_all, X_test[X_train.columns], y_test, "+CCP剪枝（完整优化）")
    results.append(m)

    df_ablation = pd.DataFrame(results)
    df_ablation.to_csv(os.path.join(OUTPUT_DIR, "ablation_study.csv"), index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(df_ablation))
    width = 0.25
    ax.bar([i - width for i in x], df_ablation["AUC"], width, label="AUC", color="steelblue")
    ax.bar(x, df_ablation["F1-Score"], width, label="F1-Score", color="darkorange")
    ax.bar([i + width for i in x], 1 - df_ablation["LogLoss"], width, label="1-LogLoss", color="green")
    ax.set_xticks(x)
    ax.set_xticklabels(df_ablation["Model"], rotation=20, fontsize=9)
    ax.set_ylabel("指标值")
    ax.set_title("消融实验结果")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "ablation_study.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return df_ablation


def save_model(model, filename="optimized_rf_model.pkl"):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[保存] 模型已保存至 {path}")
