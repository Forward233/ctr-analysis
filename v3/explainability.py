import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from config import FIGURE_DIR

# 修复: 使用 Noto Sans CJK SC 替代 SimHei，兼容 Linux 环境
# 如果在 Windows 上运行可改回 SimHei
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def shap_analysis(model, X_test, feature_names=None, max_display=20):
    if feature_names is None:
        feature_names = X_test.columns.tolist()

    sample_size = min(2000, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # 修复: 兼容不同版本 SHAP 的返回格式
    # 旧版返回 list[ndarray]，新版可能返回 3D ndarray
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_vals = shap_values[:, :, 1]
    else:
        shap_vals = shap_values

    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_vals, X_sample, feature_names=feature_names,
                      max_display=max_display, show=False)
    plt.title("SHAP特征重要性摘要图", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()

    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Mean |SHAP|": mean_abs_shap
    }).sort_values("Mean |SHAP|", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 8))
    top_n = min(max_display, len(feature_importance))
    ax.barh(
        feature_importance["Feature"].head(top_n)[::-1],
        feature_importance["Mean |SHAP|"].head(top_n)[::-1],
        color="coral"
    )
    ax.set_xlabel("平均|SHAP值|", fontsize=12)
    ax.set_title("SHAP特征重要性排序", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "shap_importance_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()

    top_features = feature_importance["Feature"].head(4).tolist()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, feat in enumerate(top_features):
        ax = axes[idx // 2][idx % 2]
        feat_idx = feature_names.index(feat)
        shap.dependence_plot(
            feat_idx, shap_vals, X_sample,
            feature_names=feature_names, ax=ax, show=False
        )
        ax.set_title(f"SHAP依赖图: {feat}", fontsize=11)
    plt.suptitle("Top 4 特征SHAP依赖图", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "shap_dependence.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[SHAP] 分析完成，图表已保存至 {FIGURE_DIR}")
    return feature_importance, shap_vals


def rf_feature_importance(model, feature_names):
    importances = model.feature_importances_
    fi = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 8))
    top_n = min(20, len(fi))
    ax.barh(fi["Feature"].head(top_n)[::-1], fi["Importance"].head(top_n)[::-1], color="teal")
    ax.set_xlabel("特征重要性 (Gini)", fontsize=12)
    ax.set_title("随机森林特征重要性 (Gini Importance)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "rf_feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return fi
