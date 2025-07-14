import numpy as np
import pandas as pd
import warnings


def map_timestamps_robust_final(
        query_timestamps_open: np.ndarray,
        large_period_open_timestamps: np.ndarray) -> np.ndarray:
    """
    将小周期K线的开盘时间戳，映射到其**之前**最近的**已收盘**的大周期K线的顺序索引。

    该函数严格遵循无前瞻性原则，并鲁棒处理以下边际情况：
    1. 输入检查：校验输入类型和维度。
    2. 大周期K线数据量不足或为空：直接抛出 ValueError。
    3. 查询时间点早于第一个大周期K线的收盘时间：映射为 np.nan。
    4. 查询时间点与大周期K线开盘/收盘时间精确重合：正确映射到已收盘K线。
    5. 大周期K线数据中存在的跳空 (通过实际下一个K线的开盘时间作为收盘边界)。
    6. 查询时间点晚于所有已知大周期K线范围，将其映射到最后一个可用大周期K线并发出警告。

    Args:
        query_timestamps_open (np.ndarray): 小周期K线（例如15分钟、1小时）的**开盘时间戳**。
                                            必须是数值型（如毫秒），且已按升序排序。
        large_period_open_timestamps (np.ndarray): 大周期K线（例如4小时、日线）的**开盘时间戳**。
                                                  必须是数值型（如毫秒），且已按升序排序。

    Returns:
        np.ndarray: 一个与 query_timestamps_open 长度相同的NumPy数组，
                    其中每个元素是 query_timestamps_open 对应元素所匹配到的
                    大周期K线的顺序索引（dtype=float64）。
                    对于无法匹配到已收盘K线的查询，返回 np.nan。

    Raises:
        TypeError: 如果输入的时间戳不是 NumPy 数组。
        ValueError: 如果输入的时间戳数组不是一维的，或者大周期K线数据不足 (少于2条)。
    """
    if not isinstance(query_timestamps_open, np.ndarray) or not isinstance(
            large_period_open_timestamps, np.ndarray):
        raise TypeError("输入的时间戳必须是 NumPy 数组。")
    if query_timestamps_open.ndim != 1 or large_period_open_timestamps.ndim != 1:
        raise ValueError("输入的时间戳数组必须是一维的。")

    # 核心修改：对大周期K线数据量不足的情况直接报错
    if large_period_open_timestamps.size < 2:
        raise ValueError(f"大周期K线数据不足，无法进行有效映射。至少需要两条大周期K线才能确定收盘边界。"
                         f"当前大周期K线数量: {large_period_open_timestamps.size}。")

    # 大周期K线的收盘边界：使用下一个大周期K线的开盘时间
    large_period_close_boundaries = large_period_open_timestamps[1:]

    # 查找每个查询时间戳在收盘边界数组中的插入位置
    raw_indices_from_searchsorted = np.searchsorted(
        large_period_close_boundaries, query_timestamps_open, side='right')

    # 初始映射：插入位置减1得到对应的大周期K线索引
    result_indices_candidate = raw_indices_from_searchsorted - 1

    # 初始化结果数组为NaN
    result_indices = np.full_like(query_timestamps_open,
                                  np.nan,
                                  dtype=np.float64)

    # 筛选有效映射：索引必须在[0, large_period_open_timestamps.size - 1]范围内
    valid_map_mask = (result_indices_candidate >= 0) & \
                     (result_indices_candidate < large_period_open_timestamps.size)

    result_indices[valid_map_mask] = result_indices_candidate[valid_map_mask]

    # 警告逻辑：查询时间点超出所有可用大周期K线的最新开盘时间
    # 此时，即使映射到最后一个K线，也说明数据可能已经滞后。
    if large_period_open_timestamps.size > 0:
        last_large_period_idx = large_period_open_timestamps.size - 1
        last_large_period_open_ts = large_period_open_timestamps[-1]

        # 修改警告条件：当映射到倒数第二个K线且查询时间超出最新开盘时间时发出警告
        late_query_mask = (result_indices == (large_period_open_timestamps.size - 2)) & \
                          (query_timestamps_open > last_large_period_open_ts) & \
                          (~np.isnan(result_indices))

        if np.any(late_query_mask):
            first_late_query_ts = query_timestamps_open[late_query_mask][0]
            warnings.warn(
                f"警告：检测到查询时间戳 {pd.to_datetime(first_late_query_ts, unit='ms')} 或之后的时间点，"
                f"已超出所有可用大周期 K 线的最新开盘时间 ({pd.to_datetime(last_large_period_open_ts, unit='ms')})。"
                f"这些查询将映射到最后一个可用大周期 K 线 (开盘时间: {pd.to_datetime(last_large_period_open_ts, unit='ms')})，"
                "这可能表示大周期数据不新鲜或有滞后。", UserWarning)

    return result_indices
