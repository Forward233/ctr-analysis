"""
优化版数据生成脚本
修复原数据生成公式的设计缺陷：
1. 让业务特征（item_ctr_history, user_category_preference, position_in_list）成为主要影响因素
2. 降低age/gender的影响权重
3. 减小噪声，提高信噪比
4. 调整特征分布，增加区分度
"""

import pandas as pd
import numpy as np
import os

RANDOM_STATE = 42


def generate_optimized_data(n_samples=100000):
    np.random.seed(RANDOM_STATE)

    # 用户特征
    user_ids = np.random.randint(1, 10001, n_samples)
    ages = np.random.choice([1, 2, 3, 4, 5, 6], n_samples, p=[0.05, 0.15, 0.30, 0.25, 0.15, 0.10])
    genders = np.random.choice([0, 1, 2], n_samples, p=[0.1, 0.45, 0.45])
    user_levels = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.3, 0.25, 0.15])

    # 商品特征
    item_ids = np.random.randint(1, 50001, n_samples)
    category_ids = np.random.randint(1, 201, n_samples)
    brand_ids = np.random.randint(1, 501, n_samples)
    prices = np.round(np.random.lognormal(4, 1, n_samples), 2)
    sales_volumes = np.random.poisson(500, n_samples)
    item_ratings = np.clip(np.random.normal(4.0, 0.8, n_samples), 1, 5).round(1)
    item_review_count = np.random.poisson(100, n_samples)

    # 时间特征
    timestamps = pd.date_range("2024-01-01", periods=n_samples, freq="2min")
    timestamps_arr = timestamps.values.copy()
    np.random.shuffle(timestamps_arr)
    timestamps = pd.DatetimeIndex(timestamps_arr)
    hours = pd.Series(timestamps).dt.hour.values
    is_weekend = pd.Series(timestamps).dt.dayofweek.isin([5, 6]).astype(int).values
    day_of_week = pd.Series(timestamps).dt.dayofweek.values
    is_holiday = np.random.choice([0, 1], n_samples, p=[0.92, 0.08])
    device_type = np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1])

    # 用户行为特征
    user_click_history = np.random.poisson(15, n_samples)
    user_purchase_history = np.random.poisson(3, n_samples)
    user_avg_dwell_time = np.round(np.random.exponential(30, n_samples), 1)
    user_category_preference = np.random.rand(n_samples).round(3)
    user_brand_preference = np.random.rand(n_samples).round(3)
    user_session_depth = np.random.poisson(5, n_samples) + 1
    
    # 优化：item_ctr_history使用更合理的分布，均值更高
    item_ctr_history = np.clip(np.random.beta(3, 7, n_samples), 0, 1).round(4)
    item_conversion_rate = np.clip(np.random.beta(2, 15, n_samples), 0, 1).round(4)
    price_rank_in_category = np.random.rand(n_samples).round(3)
    position_in_list = np.random.randint(1, 51, n_samples)

    # ==================== 优化后的点击概率公式 ====================
    # 设计原则：
    # 1. 业务相关特征权重高（item_ctr_history, user_category_preference, position_in_list）
    # 2. 人口统计特征权重低（age, gender）
    # 3. 噪声降低
    # 4. 整体概率分布更合理（均值约25%，有足够的方差）
    
    # 构造非线性交互信号，提升特征区分度
    user_item_match = user_category_preference * item_ctr_history
    
    click_prob = (
        -0.38  # 基础概率
        # 商品历史表现（核心因素）
        + 0.50 * item_ctr_history
        + 0.20 * item_conversion_rate
        + 0.15 * np.clip(item_ratings / 5, 0, 1)
        
        # 用户偏好（重要因素）
        + 0.35 * user_category_preference
        + 0.20 * user_brand_preference
        
        # 用户-商品匹配度（交互特征）
        + 0.25 * user_item_match
        
        # 位置效应
        - 0.25 * np.clip((position_in_list - 1) / 49, 0, 1)
        
        # 用户行为历史
        + 0.08 * np.clip(user_click_history / 30, 0, 1)
        
        # 时间效应
        + 0.06 * is_weekend
        + 0.08 * ((hours >= 19) & (hours <= 23)).astype(float)
        
        # 价格效应
        - 0.12 * np.clip(prices / 500, 0, 1)
        + 0.06 * (1 - price_rank_in_category)
        
        # 人口统计（次要）
        + 0.02 * (ages == 3).astype(float)
        + 0.01 * (genders == 1).astype(float)
        + 0.02 * (user_levels >= 4).astype(float)
        
        # 设备效应
        + 0.03 * (device_type == 0).astype(float)
        
        # 销量效应
        + 0.05 * np.clip(sales_volumes / 1000, 0, 1)
        
        # 噪声
        + np.random.normal(0, 0.008, n_samples)
    )
    
    # 裁剪到合理范围
    click_prob = np.clip(click_prob, 0.02, 0.95)
    
    # 生成点击标签
    click = (np.random.rand(n_samples) < click_prob).astype(int)

    # 构建DataFrame
    df = pd.DataFrame({
        "user_id": user_ids,
        "age_group": ages,
        "gender": genders,
        "user_level": user_levels,
        "item_id": item_ids,
        "category_id": category_ids,
        "brand_id": brand_ids,
        "price": prices,
        "sales_volume": sales_volumes,
        "item_rating": item_ratings,
        "item_review_count": item_review_count,
        "hour": hours,
        "is_weekend": is_weekend,
        "day_of_week": day_of_week,
        "is_holiday": is_holiday,
        "device_type": device_type,
        "user_click_history": user_click_history,
        "user_purchase_history": user_purchase_history,
        "user_avg_dwell_time": user_avg_dwell_time,
        "user_category_preference": user_category_preference,
        "user_brand_preference": user_brand_preference,
        "user_session_depth": user_session_depth,
        "item_ctr_history": item_ctr_history,
        "item_conversion_rate": item_conversion_rate,
        "price_rank_in_category": price_rank_in_category,
        "position_in_list": position_in_list,
        "click": click,
    })

    return df


def validate_data(df):
    """验证生成数据的质量"""
    print("=" * 60)
    print("数据质量验证")
    print("=" * 60)
    
    print(f"\n1. 基本统计:")
    print(f"   样本数: {len(df):,}")
    print(f"   正样本数: {df['click'].sum():,}")
    print(f"   正样本比例: {df['click'].mean():.2%}")
    
    print(f"\n2. 特征与click的相关性 (Top 10):")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()['click'].drop('click').abs().sort_values(ascending=False)
    for feat, val in corr.head(10).items():
        print(f"   {feat}: {val:.4f}")
    
    print(f"\n3. 关键特征分组点击率:")
    print(f"   age_group分组:")
    for ag in sorted(df['age_group'].unique()):
        ctr = df[df['age_group'] == ag]['click'].mean()
        print(f"     age_group={ag}: {ctr:.2%}")
    
    print(f"   gender分组:")
    for g in sorted(df['gender'].unique()):
        ctr = df[df['gender'] == g]['click'].mean()
        print(f"     gender={g}: {ctr:.2%}")
    
    print(f"\n   position_in_list分组:")
    df['pos_group'] = pd.cut(df['position_in_list'], bins=[0, 10, 20, 30, 40, 50], labels=['1-10', '11-20', '21-30', '31-40', '41-50'])
    for pg in ['1-10', '11-20', '21-30', '31-40', '41-50']:
        ctr = df[df['pos_group'] == pg]['click'].mean()
        print(f"     position {pg}: {ctr:.2%}")
    df.drop('pos_group', axis=1, inplace=True)
    
    print(f"\n   item_ctr_history分组:")
    df['ctr_group'] = pd.cut(df['item_ctr_history'], bins=[0, 0.1, 0.2, 0.3, 0.4, 1.0], labels=['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4+'])
    for cg in ['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4+']:
        subset = df[df['ctr_group'] == cg]
        if len(subset) > 0:
            ctr = subset['click'].mean()
            print(f"     item_ctr {cg}: {ctr:.2%} (n={len(subset):,})")
    df.drop('ctr_group', axis=1, inplace=True)


if __name__ == "__main__":
    import os
    print("生成优化版数据集...")
    df = generate_optimized_data(n_samples=100000)

    validate_data(df)

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, "synthetic_ecommerce_ctr.csv")
    df.to_csv(output_path, index=False)
    print(f"\n数据已保存至: {output_path}")
