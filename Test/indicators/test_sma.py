import jax.numpy as jnp
import numpy as np
import pandas as pd
import pandas_ta as ta
import pytest
from Test.conftest import df_data, np_data
from Test.test_utils import assert_indicator_same, assert_indicator_different

from src.examples.run_benchmark import run_benchmark
from src.utils.config_utils import indicator_params_template


def test_sma_accuracy(df_data,
                      np_data,
                      talib=False,
                      assert_func=assert_indicator_same):
    '''
    测试 SMA 指标的准确性, 以talib为准
    '''

    close = np_data[:, 4]
    close_series = df_data["close"]

    indicator_params = indicator_params_template(period=[14, 15, 1],
                                                 enable_list=["sma"])

    result = run_benchmark(np_data,
                           indicator_params=indicator_params,
                           enable_cpu=True,
                           enable_gpu=False,
                           enable_run_second=False)

    cpu_result = result["cpu_result"]
    sma_cpu_result = np.asarray(
        cpu_result["micro"]["indicators"]["sma"]["sma_result"])

    # 使用 pandas_ta 计算相同参数的 SMA 结果
    pandas_sma_results = []
    for period in indicator_params["sma"]["period"]:
        _ = ta.sma(close_series, length=int(period), talib=talib)
        pandas_sma_results.append(_)

    # 比较 JAX 计算的 SMA 结果和 pandas-ta 计算的 SMA 结果
    for index, period in enumerate(indicator_params["sma"]["period"]):
        assert_func(sma_cpu_result[index], pandas_sma_results[index], "sma",
                    f"period {period}")


def test_atr_accuracy_talib(df_data,
                            np_data,
                            talib=True,
                            assert_func=assert_indicator_same):
    test_sma_accuracy(df_data, np_data, talib, assert_func)


def test_pandas_ta_and_talib_sma_same(df_data):
    """
    比较 pandas-ta 和 talib 计算的 SMA 结果是否相同。
    预期结果是相同，所以使用 assert_indicator_same。
    """
    close_series = df_data["close"]

    # 定义要测试的 SMA 周期
    sma_periods_to_test = [10, 20, 30]

    for period in sma_periods_to_test:
        # 使用 pandas_ta 计算 SMA
        pandas_sma = ta.sma(close_series, length=int(period), talib=False)

        # 使用 talib 计算 SMA
        talib_sma = ta.sma(close_series, length=int(period), talib=True)

        # 使用 assert_indicator_same 验证两者是否相同
        assert_indicator_same(pandas_sma, talib_sma, "sma", f"period {period}")
