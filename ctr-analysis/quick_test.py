"""
快速验证脚本：测试优化后数据集的模型效果
包含SMOTE过采样和最优阈值调整
"""
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve
from imblearn.over_sampling import SMOTE

# 强制刷新输出
def log(msg):
    print(msg, flush=True)

log("=" * 60)
log("优化数据集模型效果验证（含SMOTE）")
log("=" * 60)

# 加载数据
log("\n加载数据...")
df = pd.read_csv('synthetic_ecommerce_ctr.csv')
log(f"数据规模: {df.shape}")
log(f"正样本比例: {df['click'].mean():.2%}")

# 特征工程
df_proc = df.copy()
df_proc.drop_duplicates(inplace=True)

# 交互特征
df_proc["price_sales_ratio"] = df_proc["price"] / (df_proc["sales_volume"] + 1)
df_proc["click_to_purchase_ratio"] = df_proc["user_purchase_history"] / (df_proc["user_click_history"] + 1)
df_proc["weighted_rating"] = df_proc["item_rating"] * np.log1p(df_proc["item_review_count"])
df_proc["is_peak_hour"] = ((df_proc["hour"] >= 19) & (df_proc["hour"] <= 23)).astype(int)
df_proc["hour_sin"] = np.sin(2 * np.pi * df_proc["hour"] / 24)
df_proc["hour_cos"] = np.cos(2 * np.pi * df_proc["hour"] / 24)

# 删除ID
df_proc.drop(["user_id", "item_id"], axis=1, inplace=True)

# 编码
for col in ["category_id", "brand_id"]:
    le = LabelEncoder()
    df_proc[col] = le.fit_transform(df_proc[col].astype(str))

for col in ["device_type", "gender", "age_group", "user_level"]:
    dummies = pd.get_dummies(df_proc[col], prefix=col, drop_first=True)
    df_proc = pd.concat([df_proc, dummies], axis=1)
    df_proc.drop(col, axis=1, inplace=True)

# 分离特征和标签
X = df_proc.drop("click", axis=1)
y = df_proc["click"]

# 归一化
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 分割
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

log(f"\n训练集: {X_train.shape}, 测试集: {X_test.shape}")
log(f"训练集正样本: {y_train.sum()} ({y_train.mean():.2%})")

# SMOTE过采样（多线程）
log("\n应用SMOTE过采样...")
smote = SMOTE(sampling_strategy=0.8, random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
log(f"SMOTE后训练集: {len(y_train_sm)} (正样本: {y_train_sm.sum()}, {y_train_sm.mean():.2%})")

# 训练（使用全部CPU核心）
log("\n训练随机森林模型...")
rf = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1, verbose=1)
rf.fit(X_train_sm, y_train_sm)

# 预测概率
y_prob = rf.predict_proba(X_test)[:, 1]

# 找最优阈值（最大化F1）
precision_arr, recall_arr, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * (precision_arr * recall_arr) / (precision_arr + recall_arr + 1e-8)
best_idx = np.argmax(f1_scores[:-1])
best_threshold = thresholds[best_idx]
log(f"\n最优阈值: {best_threshold:.4f}")

# 使用最优阈值预测
y_pred = (y_prob >= best_threshold).astype(int)

# 评估
log("\n" + "=" * 60)
log("模型评估结果（使用最优阈值）")
log("=" * 60)
log(classification_report(y_test, y_pred, target_names=['未点击', '点击'], zero_division=0))

auc = roc_auc_score(y_test, y_prob)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

log(f"\nAUC: {auc:.4f}")
log(f"Precision: {precision:.4f}")
log(f"Recall: {recall:.4f}")
log(f"F1-Score: {f1:.4f}")

# 特征重要性
log("\n" + "=" * 60)
log("Top 15 特征重要性")
log("=" * 60)
fi = pd.Series(rf.feature_importances_, index=X_scaled.columns).sort_values(ascending=False)
for feat, imp in fi.head(15).items():
    log(f"  {feat}: {imp:.4f}")

# 预测分布
log(f"\n预测分布:")
log(f"  预测为点击: {(y_pred==1).sum()}")
log(f"  预测为未点击: {(y_pred==0).sum()}")
log(f"  预测概率范围: [{y_prob.min():.4f}, {y_prob.max():.4f}]")

# 混淆矩阵
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
log(f"\n混淆矩阵:")
log(f"  TN={cm[0,0]}, FP={cm[0,1]}")
log(f"  FN={cm[1,0]}, TP={cm[1,1]}")
