import jax.numpy as jnp
import numpy as np
import pandas as pd
import pandas_ta as ta
import pytest
from Test.conftest import df_data, np_data
from Test.test_utils import assert_indicator_same, assert_indicator_different

from src.examples.run_benchmark import run_benchmark
from src.utils.config_utils import indicator_params_template


def test_rsi_accuracy(df_data,
                      np_data,
                      talib=False,
                      assert_func=assert_indicator_different):
    '''
    测试 JAX 实现的 RSI 指标的准确性，与 pandas_ta 库的结果进行比较。

    此测试用例旨在验证 JAX 版 RSI 在不同周期下计算结果的正确性，
    特别是关注其与 TA-Lib (通过 pandas_ta 包装) 行为的一致性。
    注意：TA-Lib 默认行为通常会导致前导 NaN 数量与某些纯 Python 实现不同。
    本测试以 TA-Lib 的输出为准。

    Args:
        df_data (pd.DataFrame): 包含时间序列数据的 DataFrame，用于 pandas_ta 计算。
        np_data (jnp.ndarray): 包含时间序列数据的 NumPy 数组，用于 JAX 计算。
    '''
    time = np_data[:, 0]
    open = np_data[:, 1]
    high = np_data[:, 2]
    low = np_data[:, 3]
    close = np_data[:, 4]
    volume = np_data[:, 5]

    time_series = df_data["time"]
    open_series = df_data["open"]
    high_series = df_data["high"]
    low_series = df_data["low"]
    close_series = df_data["close"]
    volume_series = df_data["volume"]

    indicator_params = indicator_params_template(period=[14, 15, 1],
                                                 enable_list=["rsi"])

    result = run_benchmark(np_data,
                           indicator_params=indicator_params,
                           enable_cpu=True,
                           enable_gpu=False,
                           enable_run_second=False)

    cpu_result = result["cpu_result"]
    rsi_cpu_result = np.asarray(
        cpu_result["micro"]["indicators"]["rsi"]["rsi_result"])

    # 使用 pandas_ta 计算相同参数的 RSI 结果
    pandas_rsi_results = []
    for period in indicator_params["rsi"]["period"]:
        _ = ta.rsi(close_series, length=int(period), talib=talib)
        pandas_rsi_results.append(_)

    # 比较 JAX 计算的 RSI 结果和 pandas-ta 计算的 RSI 结果
    for index, period in enumerate(indicator_params["rsi"]["period"]):
        assert_func(rsi_cpu_result[index], pandas_rsi_results[index], "rsi",
                    f"period {period}")


def test_atr_accuracy_talib(df_data,
                            np_data,
                            talib=True,
                            assert_func=assert_indicator_same):
    test_rsi_accuracy(df_data, np_data, talib, assert_func)


def test_pandas_ta_and_talib_rsi_difference(df_data):
    """
    比较 pandas-ta 和 talib 计算的 RSI 结果是否不同。
    预期结果是不同，所以使用 assert_indicator_different。
    """
    close_series = df_data["close"]
    close_np = close_series.to_numpy()  # TA-Lib 通常需要 NumPy 数组

    # 定义要测试的 RSI 周期，可以与上面的 JAX 测试使用相同的周期
    rsi_periods_to_test = [10, 20, 30]

    for period in rsi_periods_to_test:
        # 使用 pandas_ta 计算 RSI
        pandas_rsi = ta.rsi(close_series, length=int(period), talib=False)

        # 使用 talib 计算 RSI
        talib_rsi = ta.rsi(close_series, timeperiod=int(period), talib=True)

        # 使用 assert_indicator_different 验证两者是否不同
        # 传递的参数名称保持一致，方便追踪
        assert_indicator_different(pandas_rsi, talib_rsi, "rsi",
                                   f"period {period} (Pandas-TA vs TA-Lib)")
