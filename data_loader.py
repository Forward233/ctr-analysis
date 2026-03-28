import pandas as pd
import numpy as np
from config import DATA_DIR, RANDOM_STATE
import os


def _unused_generate_synthetic_data(n_samples=100000):
    """
    优化版数据生成函数
    设计原则：
    1. 业务特征（item_ctr_history, user_category_preference, position_in_list）为主要影响因素
    2. 人口统计特征（age, gender）为次要因素
    3. 低噪声，高信噪比
    """
    np.random.seed(RANDOM_STATE)

    user_ids = np.random.randint(1, 10001, n_samples)
    ages = np.random.choice([1, 2, 3, 4, 5, 6], n_samples, p=[0.05, 0.15, 0.30, 0.25, 0.15, 0.10])
    genders = np.random.choice([0, 1, 2], n_samples, p=[0.1, 0.45, 0.45])
    user_levels = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.3, 0.25, 0.15])

    item_ids = np.random.randint(1, 50001, n_samples)
    category_ids = np.random.randint(1, 201, n_samples)
    brand_ids = np.random.randint(1, 501, n_samples)
    prices = np.round(np.random.lognormal(4, 1, n_samples), 2)
    sales_volumes = np.random.poisson(500, n_samples)
    item_ratings = np.clip(np.random.normal(4.0, 0.8, n_samples), 1, 5).round(1)
    item_review_count = np.random.poisson(100, n_samples)

    timestamps = pd.date_range("2024-01-01", periods=n_samples, freq="2min")
    timestamps_arr = timestamps.values.copy()
    np.random.shuffle(timestamps_arr)
    timestamps = pd.DatetimeIndex(timestamps_arr)
    hours = pd.Series(timestamps).dt.hour.values
    is_weekend = pd.Series(timestamps).dt.dayofweek.isin([5, 6]).astype(int).values
    day_of_week = pd.Series(timestamps).dt.dayofweek.values
    is_holiday = np.random.choice([0, 1], n_samples, p=[0.92, 0.08])
    device_type = np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1])

    user_click_history = np.random.poisson(15, n_samples)
    user_purchase_history = np.random.poisson(3, n_samples)
    user_avg_dwell_time = np.round(np.random.exponential(30, n_samples), 1)
    user_category_preference = np.random.rand(n_samples).round(3)
    user_brand_preference = np.random.rand(n_samples).round(3)
    user_session_depth = np.random.poisson(5, n_samples) + 1
    item_ctr_history = np.clip(np.random.beta(3, 7, n_samples), 0, 1).round(4)
    item_conversion_rate = np.clip(np.random.beta(2, 15, n_samples), 0, 1).round(4)
    price_rank_in_category = np.random.rand(n_samples).round(3)
    position_in_list = np.random.randint(1, 51, n_samples)

    user_item_match = user_category_preference * item_ctr_history

    click_prob = (
        -0.38
        + 0.50 * item_ctr_history
        + 0.20 * item_conversion_rate
        + 0.15 * np.clip(item_ratings / 5, 0, 1)
        + 0.35 * user_category_preference
        + 0.20 * user_brand_preference
        + 0.25 * user_item_match
        - 0.25 * np.clip((position_in_list - 1) / 49, 0, 1)
        + 0.08 * np.clip(user_click_history / 30, 0, 1)
        + 0.06 * is_weekend
        + 0.08 * ((hours >= 19) & (hours <= 23)).astype(float)
        - 0.12 * np.clip(prices / 500, 0, 1)
        + 0.06 * (1 - price_rank_in_category)
        + 0.02 * (ages == 3).astype(float)
        + 0.01 * (genders == 1).astype(float)
        + 0.02 * (user_levels >= 4).astype(float)
        + 0.03 * (device_type == 0).astype(float)
        + 0.05 * np.clip(sales_volumes / 1000, 0, 1)
        + np.random.normal(0, 0.008, n_samples)
    )
    click_prob = np.clip(click_prob, 0.02, 0.95)
    click = (np.random.rand(n_samples) < click_prob).astype(int)

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


def load_data(data_path=None):
    path = data_path if data_path else os.path.join(DATA_DIR, "ecommerce_ctr.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据文件不存在: {path}")
    return pd.read_csv(path)
