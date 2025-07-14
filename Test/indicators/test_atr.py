import jax.numpy as jnp
import numpy as np
import pandas as pd
import pandas_ta as ta
import pytest
from Test.conftest import df_data, np_data
from Test.test_utils import assert_indicator_same, assert_indicator_different

from src.examples.run_benchmark import run_benchmark
from src.utils.config_utils import indicator_params_template


def test_atr_accuracy(df_data,
                      np_data,
                      talib=False,
                      assert_func=assert_indicator_different):
    '''
    测试 ATR 指标的准确性, 以talib为准
    '''

    high = np_data[:, 2]
    low = np_data[:, 3]
    close = np_data[:, 4]

    high_series = df_data["high"]
    low_series = df_data["low"]
    close_series = df_data["close"]

    indicator_params = indicator_params_template(period=[14, 15, 1],
                                                 enable_list=["atr"])

    result = run_benchmark(np_data,
                           indicator_params=indicator_params,
                           enable_cpu=True,
                           enable_gpu=False,
                           enable_run_second=False)

    cpu_result = result["cpu_result"]
    # atr_jax 返回的是 (atr_values, tr_values)，所以需要提取 atr_values
    atr_cpu_result = np.asarray(
        cpu_result["micro"]["indicators"]["atr"]["atr_result"])

    # 使用 pandas_ta 计算相同参数的 ATR 结果
    pandas_atr_results = []
    for period in indicator_params["atr"]["period"]:
        _ = ta.atr(high_series,
                   low_series,
                   close_series,
                   length=int(period),
                   talib=talib)
        pandas_atr_results.append(_)

    # 比较 JAX 计算的 ATR 结果和 pandas-ta 计算的 ATR 结果
    for index, period in enumerate(indicator_params["atr"]["period"]):
        assert_func(atr_cpu_result[index], pandas_atr_results[index], "atr",
                    f"period {period}")


def test_atr_accuracy_talib(df_data,
                            np_data,
                            talib=True,
                            assert_func=assert_indicator_same):
    test_atr_accuracy(df_data, np_data, talib, assert_func)


def test_pandas_ta_and_talib_atr_difference(df_data):
    """
    比较 pandas-ta 和 talib 计算的 ATR 结果是否不同。
    预期结果是不同，所以使用 assert_indicator_different。
    """
    high_series = df_data["high"]
    low_series = df_data["low"]
    close_series = df_data["close"]

    # 定义要测试的 ATR 周期
    atr_periods_to_test = [10, 20, 30]

    for period in atr_periods_to_test:
        # 使用 pandas_ta 计算 ATR (talib=False)
        pandas_ta_atr = ta.atr(high_series,
                               low_series,
                               close_series,
                               length=int(period),
                               talib=False)

        # 使用 talib 计算 ATR (talib=True)
        talib_atr = ta.atr(high_series,
                           low_series,
                           close_series,
                           length=int(period),
                           talib=True)

        # 使用 assert_indicator_different 验证两者是否不同
        assert_indicator_different(pandas_ta_atr, talib_atr, "atr",
                                   f"period {period} (Pandas-TA vs TA-Lib)")
