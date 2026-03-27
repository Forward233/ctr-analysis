import pandas as pd
import numpy as np

df = pd.read_csv('synthetic_ecommerce_ctr.csv')

print('=== 连续特征分布 ===')
for col in ['item_ctr_history', 'user_category_preference', 'user_brand_preference', 'price', 'user_click_history']:
    print(f'{col}: min={df[col].min():.4f}, max={df[col].max():.4f}, mean={df[col].mean():.4f}')

print('\n=== 与click的皮尔逊相关系数 ===')
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()['click'].drop('click').abs().sort_values(ascending=False)
print(corr.head(15))

print('\n=== 数据生成公式中各因素的实际贡献范围 ===')
# 模拟公式中各项的贡献
contrib = {
    'age_group=3': 0.08 * (df['age_group'] == 3).astype(float),
    'gender=1': 0.05 * (df['gender'] == 1).astype(float),
    'item_ctr_history': 0.10 * df['item_ctr_history'],
    'user_category_preference': 0.06 * df['user_category_preference'],
    'user_brand_preference': 0.04 * df['user_brand_preference'],
    'user_click_history': 0.03 * np.clip(df['user_click_history'] / 30, 0, 1),
    'price': -0.05 * np.clip(df['price'] / 1000, 0, 1),
    'item_rating': 0.04 * np.clip(df['item_rating'] / 5, 0, 1),
    'is_weekend': 0.03 * df['is_weekend'],
    'is_peak_hour': 0.05 * ((df['hour'] >= 19) & (df['hour'] <= 23)).astype(float),
    'position_in_list': -0.03 * np.clip(df['position_in_list'] / 50, 0, 1),
}

for name, values in contrib.items():
    print(f'{name}: mean={values.mean():.4f}, std={values.std():.4f}, range=[{values.min():.4f}, {values.max():.4f}]')
