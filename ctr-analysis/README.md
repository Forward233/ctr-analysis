# 基于优化随机森林的电商商品推荐点击率预测研究

## 项目简介

本项目实现了一个基于优化随机森林的电商CTR（点击率）预测系统，采用多种优化策略提升模型性能。

## 主要特性

- **LASSO特征筛选**：使用LASSO回归进行特征选择
- **SMOTE重采样**：处理类别不平衡问题
- **贝叶斯优化**：自动调优随机森林超参数
- **CCP剪枝**：防止过拟合
- **SHAP可解释性分析**：模型解释

## 环境配置

```bash
# 创建conda环境
conda create -n ctr-analysis python=3.12 -y
conda activate ctr-analysis

# 安装依赖
pip install -r requirements.txt
```

## 运行实验

```bash
python main.py
```


## 项目结构

```
├── config.py              # 配置参数（超参数搜索空间、SMOTE策略等）
├── data_loader.py         # 数据加载
├── feature_engineering.py # 特征工程、归一化、LASSO筛选
├── model.py               # 模型训练、贝叶斯优化、CCP剪枝、消融实验
├── explainability.py      # SHAP可解释性分析
├── quick_test.py          # 快速调试脚本
├── main.py                # 主程序入口
├── requirements.txt       # 依赖清单
├── 论文问题说明.md         # 代码问题记录与修正说明（共14处）
├── data/                  # 数据目录
└── output/                # 实验结果输出
    ├── figures/           # 图表（ROC、SHAP、消融实验等）
    ├── ablation_study.csv # 消融实验结果
    ├── model_comparison.csv # 模型对比结果
    └── optimized_rf_model.pkl # 最终模型（gitignore）
```

## 实验结果

### 消融实验

| 步骤 | AUC | Recall | F1 |
|------|-----|--------|-----|
| 基线RF | 0.7492 | 0.063 | 0.115 |
| +LASSO特征筛选 | 0.7513 | 0.075 | 0.136 |
| +SMOTE重采样 | 0.7412 | 0.169 | 0.259 |
| +贝叶斯优化 | 0.7466 | 0.230 | 0.323 |
| +CCP剪枝 | **0.7468** | **0.230** | **0.323** |

### 关键超参数（贝叶斯优化结果）

| 参数 | 搜索范围 | 最优值 |
|------|---------|--------|
| n_estimators | [50, 500] | 218 |
| max_depth | [3, 15] | 14 |
| min_samples_split | [2, 20] | 15 |
| min_samples_leaf | [1, 3] | 2 |
| max_features | [0.1, 0.9] | 0.225 |

## 依赖

- Python >= 3.10
- scikit-learn >= 1.3.0
- xgboost >= 2.0.0
- lightgbm >= 4.0.0
- shap >= 0.42.0
- bayesian-optimization >= 1.4.0
- imbalanced-learn >= 0.11.0

## License

MIT
