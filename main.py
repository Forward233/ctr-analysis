import sys
import time
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

from config import RANDOM_STATE, TEST_SIZE, OUTPUT_DIR, FIGURE_DIR
from data_loader import load_data
from feature_engineering import (
    preprocess_data, build_features, encode_features,
    pearson_correlation_analysis, plot_feature_distributions,
    lasso_feature_selection
)
from model import (
    apply_smote, train_baseline_rf, bayesian_optimize_rf,
    apply_ccp_pruning, evaluate_model, train_comparison_models,
    plot_model_comparison, plot_roc_curves, plot_confusion_matrix,
    run_ablation_study, save_model
)
from explainability import shap_analysis, rf_feature_importance


def main(data_path=None):
    print("=" * 70)
    print("  基于优化随机森林的电商商品推荐点击率预测研究")
    print("=" * 70)
    start_time = time.time()

    print("\n[1/9] 数据加载...")
    df = load_data(data_path)
    print(f"  数据规模: {df.shape[0]} 条记录, {df.shape[1]} 个字段")
    print(f"  正样本比例: {df['click'].mean():.4f}")

    # 生成表3-1 数据集基本统计信息
    stats = {
        "统计指标": ["样本总数", "特征字段数（原始）", "正样本数（点击）", "负样本数（未点击）",
                     "正样本比例", "用户数量", "商品数量", "品类数量", "品牌数量"],
        "数值": [
            f"{df.shape[0]:,}",
            str(df.shape[1] - 1),
            f"{int(df['click'].sum()):,}",
            f"{int((df['click'] == 0).sum()):,}",
            f"{df['click'].mean():.2%}",
            f"{df['user_id'].nunique():,}" if 'user_id' in df.columns else "N/A",
            f"{df['item_id'].nunique():,}" if 'item_id' in df.columns else "N/A",
            f"{df['category_id'].nunique():,}" if 'category_id' in df.columns else "N/A",
            f"{df['brand_id'].nunique():,}" if 'brand_id' in df.columns else "N/A",
        ]
    }
    pd.DataFrame(stats).to_csv(os.path.join(OUTPUT_DIR, "data_statistics.csv"), index=False)
    print(f"  数据统计信息已保存至 {OUTPUT_DIR}/data_statistics.csv")

    print("\n[2/9] 数据预处理...")
    df = preprocess_data(df)
    print(f"  预处理后: {df.shape[0]} 条记录")

    print("\n[3/9] 特征工程...")
    df = build_features(df)
    plot_feature_distributions(df, target_col="click")
    print(f"  构建特征后: {df.shape[1]} 个字段")

    print("\n[4/9] 特征编码与归一化...")
    df_encoded, scaler = encode_features(df, target_col="click")

    X = df_encoded.drop("click", axis=1)
    y = df_encoded["click"]
    print(f"  特征维度: {X.shape[1]}")

    print("\n  相关性分析...")
    top_corr = pearson_correlation_analysis(df_encoded, target_col="click")

    print("\n  LASSO特征筛选...")
    selected_features, lasso_model = lasso_feature_selection(X, y)
    X_selected = X[selected_features]
    print(f"  筛选后特征数: {len(selected_features)}")

    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train = X_train_full[selected_features]
    X_test = X_test_full[selected_features]

    print("\n[5/9] SMOTE重采样...")
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)

    print("\n[6/9] 模型训练...")
    print("  训练基线随机森林...")
    baseline_rf = train_baseline_rf(X_train, y_train)
    baseline_metrics, _, baseline_prob = evaluate_model(baseline_rf, X_test, y_test, "基线RF")
    print(f"  基线RF AUC: {baseline_metrics['AUC']:.4f}")

    print("\n  贝叶斯优化随机森林...")
    optimized_rf, best_params, optimizer = bayesian_optimize_rf(X_train_sm, y_train_sm)
    opt_metrics, _, opt_prob = evaluate_model(optimized_rf, X_test, y_test, "贝叶斯优化RF")
    print(f"  优化RF AUC: {opt_metrics['AUC']:.4f}")

    print("\n  CCP剪枝...")
    final_rf, ccp_alpha = apply_ccp_pruning(X_train_sm, y_train_sm, X_test, y_test, best_params)
    final_metrics, final_pred, final_prob = evaluate_model(final_rf, X_test, y_test, "优化随机森林（完整）")
    print(f"  最终模型 AUC: {final_metrics['AUC']:.4f}")

    print("\n[7/9] 对比模型训练...")
    comparison_results, model_probs = train_comparison_models(X_train_sm, y_train_sm, X_test, y_test)

    all_results = [baseline_metrics, opt_metrics, final_metrics] + comparison_results
    plot_model_comparison(all_results)
    plot_roc_curves(y_test, model_probs, opt_prob, final_prob)
    plot_confusion_matrix(y_test, final_pred, "优化随机森林")

    print("\n[8/9] 消融实验...")
    ablation_df = run_ablation_study(
        X_train, y_train, X_test_full, y_test, best_params,
        selected_features, X_train_full, ccp_alpha
    )
    print(ablation_df.to_string(index=False))

    print("\n[9/9] 可解释性分析...")
    fi_rf = rf_feature_importance(final_rf, selected_features)
    fi_shap, shap_vals = shap_analysis(final_rf, X_test, selected_features)

    save_model(final_rf)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("  实验完成！")
    print(f"  总耗时: {elapsed:.1f} 秒")
    print(f"  结果保存目录: {OUTPUT_DIR}")
    print(f"  图表保存目录: {FIGURE_DIR}")
    print("=" * 70)

    print("\n[最终模型性能汇总]")
    for k, v in final_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    return final_rf, all_results


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(data_path)
