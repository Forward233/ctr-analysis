import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LassoCV
from config import RANDOM_STATE, LASSO_ALPHA_RANGE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import FIGURE_DIR

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def preprocess_data(df):
    df = df.copy()
    df.drop_duplicates(inplace=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)

    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    return df


def build_features(df):
    df = df.copy()

    if "price" in df.columns and "sales_volume" in df.columns:
        df["price_sales_ratio"] = df["price"] / (df["sales_volume"] + 1)
    if "user_click_history" in df.columns and "user_purchase_history" in df.columns:
        df["click_to_purchase_ratio"] = df["user_purchase_history"] / (df["user_click_history"] + 1)
    if "item_rating" in df.columns and "item_review_count" in df.columns:
        df["weighted_rating"] = df["item_rating"] * np.log1p(df["item_review_count"])
    if "hour" in df.columns:
        df["is_peak_hour"] = ((df["hour"] >= 19) & (df["hour"] <= 23)).astype(int)
        df["is_morning"] = ((df["hour"] >= 6) & (df["hour"] <= 10)).astype(int)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    if "day_of_week" in df.columns:
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    if "price" in df.columns and "category_id" in df.columns:
        cat_mean_price = df.groupby("category_id")["price"].transform("mean")
        df["price_vs_category_avg"] = df["price"] / (cat_mean_price + 1)
    if "item_ctr_history" in df.columns and "item_conversion_rate" in df.columns:
        df["ctr_cvr_ratio"] = df["item_ctr_history"] / (df["item_conversion_rate"] + 0.001)

    return df


def encode_features(df, target_col="click"):
    df = df.copy()
    id_cols = ["user_id", "item_id"]
    high_cardinality_cols = ["category_id", "brand_id"]

    for col in id_cols:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    for col in high_cardinality_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    cat_cols = ["device_type", "gender", "age_group", "user_level"]
    for col in cat_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df.drop(col, axis=1, inplace=True)

    scaler = MinMaxScaler()
    continuous_cols = [
        "price", "sales_volume", "item_rating", "item_review_count",
        "user_click_history", "user_purchase_history", "user_avg_dwell_time",
        "user_category_preference", "user_brand_preference", "user_session_depth",
        "item_ctr_history", "item_conversion_rate", "price_rank_in_category",
        "position_in_list", "price_sales_ratio", "click_to_purchase_ratio",
        "weighted_rating", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "price_vs_category_avg", "ctr_cvr_ratio"
    ]
    existing_cont = [c for c in continuous_cols if c in df.columns]
    df[existing_cont] = scaler.fit_transform(df[existing_cont])

    return df, scaler


def pearson_correlation_analysis(df, target_col="click", top_n=20):
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()

    target_corr = corr_matrix[target_col].drop(target_col).abs().sort_values(ascending=False)
    top_features = target_corr.head(top_n)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    top_cols = list(top_features.index[:15]) + [target_col]
    sns.heatmap(
        numeric_df[top_cols].corr(),
        annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, ax=axes[0], square=True,
        annot_kws={"size": 7}
    )
    axes[0].set_title("特征相关性热力图 (Top 15)", fontsize=14)
    axes[0].tick_params(axis="both", labelsize=8)

    top_features.head(15).plot(kind="barh", ax=axes[1], color="steelblue")
    axes[1].set_xlabel("皮尔逊相关系数（绝对值）", fontsize=12)
    axes[1].set_title("特征与CTR的相关性排序", fontsize=14)
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "correlation_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return top_features


def plot_feature_distributions(df, features=None, target_col="click"):
    if features is None:
        features = [
            "price", "item_rating", "item_ctr_history", "user_click_history",
            "position_in_list", "user_avg_dwell_time", "sales_volume", "user_session_depth"
        ]
    features = [f for f in features if f in df.columns]
    n_features = len(features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        df[df[target_col] == 0][feat].hist(
            bins=30, alpha=0.5, label="未点击", ax=axes[i], color="blue", density=True
        )
        df[df[target_col] == 1][feat].hist(
            bins=30, alpha=0.5, label="点击", ax=axes[i], color="red", density=True
        )
        axes[i].set_title(feat, fontsize=11)
        axes[i].legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("特征分布直方图（按点击/未点击分组）", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "feature_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close()


def lasso_feature_selection(X, y):
    lasso = LassoCV(alphas=LASSO_ALPHA_RANGE, cv=5, random_state=RANDOM_STATE, max_iter=10000)
    lasso.fit(X, y)

    feature_importance = pd.Series(np.abs(lasso.coef_), index=X.columns)
    selected_features = feature_importance[feature_importance > 0].sort_values(ascending=False)

    print(f"[LASSO] 最优alpha: {lasso.alpha_:.4f}")
    print(f"[LASSO] 选择特征数: {len(selected_features)} / {X.shape[1]}")

    fig, ax = plt.subplots(figsize=(12, 6))
    selected_features.head(20).plot(kind="barh", ax=ax, color="darkorange")
    ax.set_xlabel("LASSO系数（绝对值）")
    ax.set_title("LASSO特征筛选结果 (Top 20)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "lasso_feature_selection.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return list(selected_features.index), lasso
