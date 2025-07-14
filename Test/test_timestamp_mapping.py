import numpy as np
import pandas as pd
import pytest
import warnings
from src.utils.timestamp_mapping import map_timestamps_robust_final


def test_scenario_1_hourly_to_daily_mapping():
    """
    场景 1: (1小时, 日线) - 包含周末跳空和各种边际情况的映射测试。
    """
    daily_open_datetimes = pd.to_datetime([
        '2025-07-03 00:00:00',  # Index 0 (Thu) -> 2025-07-04 00:00:00 close
        '2025-07-04 00:00:00',  # Index 1 (Fri) -> 2025-07-07 00:00:00 close (gap)
        '2025-07-07 00:00:00',  # Index 2 (Mon) -> 2025-07-08 00:00:00 close
        '2025-07-08 00:00:00',  # Index 3 (Tue) -> Last K-line in dataset
    ])
    daily_open_timestamps_ms = np.array(
        [t.value // 10**6 for t in daily_open_datetimes])

    one_hour_open_datetimes = pd.to_datetime([
        '2025-07-02 10:00:00',  # (Query 1) **Too early**, expected nan
        '2025-07-03 00:00:00',  # (Query 2) 0-day K-line just opened, expected nan (day K-line not closed)
        '2025-07-03 12:00:00',  # (Query 3) 0-day K-line in progress, expected nan
        '2025-07-03 23:59:00',  # (Query 4) 0-day K-line about to close, expected nan
        '2025-07-04 00:00:00',  # (Query 5) 0-day K-line just closed, expected **0.0**
        '2025-07-04 10:00:00',  # (Query 6) 0-day K-line closed, 1-day K-line in progress, expected **0.0**
        '2025-07-05 00:00:00',  # (Query 7) Saturday, 1-day K-line in progress, expected **0.0**
        '2025-07-06 12:00:00',  # (Query 8) Sunday, 1-day K-line in progress, expected **0.0**
        '2025-07-06 23:59:00',  # (Query 9) Before Sunday midnight, 1-day K-line in progress, expected **0.0**
        '2025-07-07 00:00:00',  # (Query 10) Monday open, 1-day K-line just closed, expected **1.0**
        '2025-07-07 10:00:00',  # (Query 11) 1-day K-line closed, 2-day K-line in progress, expected **1.0**
        '2025-07-07 23:59:00',  # (Query 12) 2-day K-line about to close, expected **1.0**
        '2025-07-08 00:00:00',  # (Query 13) Tuesday open, 2-day K-line just closed, expected **2.0**
        '2025-07-08 05:00:00',  # (Query 14) 3-day K-line in progress, expected **3.0** (warning) - corresponds to 2025-07-08 00:00:00
        '2025-07-09 00:00:00',  # (Query 15) Beyond all available daily K-line range, expected **3.0** (warning)
        '2025-07-09 10:00:00',  # (Query 16) Even later, expected **3.0** (warning)
    ])
    query_timestamps_1hr_ms = np.array(
        [t.value // 10**6 for t in one_hour_open_datetimes])

    expected_indices = np.array([
        np.nan, np.nan, np.nan, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        2.0, 2.0, 2.0, 2.0
    ])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mapped_indices = map_timestamps_robust_final(query_timestamps_1hr_ms,
                                                     daily_open_timestamps_ms)

        # Assert warnings for queries 14, 15, 16
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "警告：检测到查询时间戳 2025-07-08 05:00:00 或之后的时间点" in str(w[-1].message)

    assert np.allclose(mapped_indices, expected_indices, equal_nan=True)


def test_scenario_2_15min_to_4hr_mapping():
    """
    场景 2: (15分钟, 4小时) - 包含非交易时段和数据不足的映射测试。
    """
    four_hour_open_datetimes = pd.to_datetime([
        '2025-07-10 00:00:00',  # Index 0
        '2025-07-10 04:00:00',  # Index 1
        '2025-07-10 08:00:00',  # Index 2
        '2025-07-10 12:00:00',  # Index 3
        '2025-07-10 16:00:00',  # Index 4
        '2025-07-10 20:00:00',  # Index 5 -> Last K-line in dataset
    ])
    four_hour_open_timestamps_ms = np.array(
        [t.value // 10**6 for t in four_hour_open_datetimes])

    fifteen_min_open_datetimes = pd.to_datetime([
        '2025-07-09 23:00:00',  # (Query 17) **Too early**, expected nan
        '2025-07-10 00:00:00',  # (Query 18) 0-4H K-line just opened, expected nan
        '2025-07-10 03:59:00',  # (Query 19) 0-4H K-line about to close, expected nan
        '2025-07-10 04:00:00',  # (Query 20) 0-4H K-line just closed, expected **0.0**
        '2025-07-10 07:00:00',  # (Query 21) 0-4H K-line closed, 1-4H K-line in progress, expected **0.0**
        '2025-07-10 08:00:00',  # (Query 22) 1-4H K-line just closed, expected **1.0**
        '2025-07-10 11:00:00',  # (Query 23) 1-4H K-line closed, 2-4H K-line in progress, expected **1.0**
        '2025-07-10 20:00:00',  # (Query 24) 4-4H K-line just closed, expected **4.0**
        '2025-07-10 21:00:00',  # (Query 25) 5-4H K-line in progress, expected **5.0** (warning) - corresponds to 2025-07-10 20:00:00
        '2025-07-11 00:00:00',  # (Query 26) Beyond range, expected **5.0** (warning)
    ])
    fifteen_min_open_timestamps_ms = np.array(
        [t.value // 10**6 for t in fifteen_min_open_datetimes])

    expected_indices = np.array(
        [np.nan, np.nan, np.nan, 0.0, 0.0, 1.0, 1.0, 4.0, 4.0, 4.0])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mapped_indices = map_timestamps_robust_final(
            fifteen_min_open_timestamps_ms, four_hour_open_timestamps_ms)

        # Assert warnings for queries 25, 26
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "警告：检测到查询时间戳 2025-07-10 21:00:00 或之后的时间点" in str(w[-1].message)

    assert np.allclose(mapped_indices, expected_indices, equal_nan=True)


def test_scenario_3_insufficient_large_period_data_single_bar():
    """
    场景 3: (15分钟, 4小时) - 大周期K线数据不足 (仅一条)。
    """
    single_four_hour_open_datetimes = pd.to_datetime([
        '2025-07-12 00:00:00',  # Index 0
    ])
    single_four_hour_open_timestamps_ms = np.array(
        [t.value // 10**6 for t in single_four_hour_open_datetimes])

    query_for_single_bar = pd.to_datetime([
        '2025-07-11 23:00:00',
        '2025-07-12 00:00:00',
        '2025-07-12 01:00:00',
        '2025-07-13 00:00:00',
    ])
    query_for_single_bar_ms = np.array(
        [t.value // 10**6 for t in query_for_single_bar])

    with pytest.raises(ValueError) as excinfo:
        map_timestamps_robust_final(query_for_single_bar_ms,
                                    single_four_hour_open_timestamps_ms)
    assert "大周期K线数据不足，无法进行有效映射。至少需要两条大周期K线才能确定收盘边界。" in str(excinfo.value)


def test_scenario_4_insufficient_large_period_data_empty_array():
    """
    场景 4: (15分钟, 4小时) - 大周期K线数据不足 (空数组)。
    """
    empty_four_hour_open_timestamps_ms = np.array([], dtype=np.int64)

    query_for_empty_bar = pd.to_datetime([
        '2025-07-14 00:00:00',
        '2025-07-14 01:00:00',
    ])
    query_for_empty_bar_ms = np.array(
        [t.value // 10**6 for t in query_for_empty_bar])

    with pytest.raises(ValueError) as excinfo:
        map_timestamps_robust_final(query_for_empty_bar_ms,
                                    empty_four_hour_open_timestamps_ms)
    assert "大周期K线数据不足，无法进行有效映射。至少需要两条大周期K线才能确定收盘边界。" in str(excinfo.value)
