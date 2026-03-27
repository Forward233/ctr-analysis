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

### 方式1：Jupyter Notebook
```bash
jupyter notebook 毕业论文实验.ipynb
```

### 方式2：命令行
```bash
python main.py
```

## 项目结构

```
├── config.py              # 配置参数
├── data_loader.py         # 数据加载与合成
├── feature_engineering.py # 特征工程
├── model.py               # 模型训练与评估
├── explainability.py      # 可解释性分析
├── main.py                # 主程序入口
├── 毕业论文实验.ipynb      # 实验Notebook
├── requirements.txt       # 依赖清单
└── data/                  # 数据目录
```

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
