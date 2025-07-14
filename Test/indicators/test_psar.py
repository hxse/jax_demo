import itertools
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pandas_ta as ta
import pytest
from Test.conftest import df_data, np_data
from Test.test_utils import assert_indicator_same, assert_indicator_different

from src.examples.run_benchmark import run_benchmark
from src.utils.config_utils import indicator_params_template


def test_psar_accuracy(df_data,
                       np_data,
                       talib=False,
                       assert_func=assert_indicator_same):
    '''
    测试 PSAR 指标的准确性, 以 pandas_ta 为准
    '''
    high = np_data[:, 2]
    low = np_data[:, 3]
    close = np_data[:, 4]

    high_series = df_data["high"]
    low_series = df_data["low"]
    close_series = df_data["close"]

    indicator_params = indicator_params_template(af0=[0.02, 0.03, 0.01],
                                                 af_step=[0.02, 0.03, 0.01],
                                                 max_af=[0.2, 0.3, 0.1],
                                                 enable_list=["psar"])

    result = run_benchmark(np_data,
                           indicator_params=indicator_params,
                           enable_cpu=True,
                           enable_gpu=False,
                           enable_run_second=False)

    psar_result_data = result["cpu_result"]["micro"]["indicators"]["psar"]
    cpu_psar_results = []
    for i in range(len(indicator_params["psar"]["af0"])):
        # psar_jax 返回的是 (psar_values, psar_direction)，所以需要提取 psar_values
        psar_long_cpu_result = np.asarray(
            psar_result_data["psar_long_result"][i])
        psar_short_cpu_result = np.asarray(
            psar_result_data["psar_short_result"][i])
        psar_af_cpu_result = np.asarray(psar_result_data["psar_af_result"][i])
        psar_reversal_cpu_result = np.asarray(
            psar_result_data["psar_reversal_result"][i])
        cpu_psar_results.append((psar_long_cpu_result, psar_short_cpu_result,
                                 psar_af_cpu_result, psar_reversal_cpu_result))

    # 使用 pandas_ta 计算相同参数的 PSAR 结果
    pandas_psar_results = []
    for af0_val, af_step_val, max_af_val in zip(
            indicator_params["psar"]["af0"],
            indicator_params["psar"]["af_step"],
            indicator_params["psar"]["max_af"]):
        df_psar = ta.psar(high_series,
                          low_series,
                          close_series,
                          af0=af0_val,
                          af=af_step_val,
                          max_af=max_af_val,
                          talib=talib)
        # 修复键名构建问题，使其包含参数
        psarl_key = f"PSARl_{af0_val}_{max_af_val}"
        psars_key = f"PSARs_{af0_val}_{max_af_val}"
        psaraf_key = f"PSARaf_{af0_val}_{max_af_val}"
        psarr_key = f"PSARr_{af0_val}_{max_af_val}"
        pandas_psar_results.append((df_psar[psarl_key], df_psar[psars_key],
                                    df_psar[psaraf_key], df_psar[psarr_key]))

    # 比较 JAX 计算的 PSAR 结果和 pandas-ta 计算的 PSAR 结果
    # PSAR 的参数组合是 af0 * af_step * max_af
    param_combinations = list(
        zip(indicator_params["psar"]["af0"],
            indicator_params["psar"]["af_step"],
            indicator_params["psar"]["max_af"]))

    for index, (af0_val, af_step_val,
                max_af_val) in enumerate(param_combinations):
        assert_func(
            cpu_psar_results[index][0], pandas_psar_results[index][0],
            "psar_long",
            f"af0 {af0_val}, af_step {af_step_val}, max_af {max_af_val}")
        assert_func(
            cpu_psar_results[index][1], pandas_psar_results[index][1],
            "psar_short",
            f"af0 {af0_val}, af_step {af_step_val}, max_af {max_af_val}")
        assert_func(
            cpu_psar_results[index][2], pandas_psar_results[index][2],
            "psar_af",
            f"af0 {af0_val}, af_step {af_step_val}, max_af {max_af_val}")
        assert_func(
            cpu_psar_results[index][3], pandas_psar_results[index][3],
            "psar_reversal",
            f"af0 {af0_val}, af_step {af_step_val}, max_af {max_af_val}")


def test_psar_accuracy_talib(df_data,
                             np_data,
                             talib=True,
                             assert_func=assert_indicator_same):
    test_psar_accuracy(df_data, np_data, talib, assert_func)


def test_pandas_ta_and_talib_psar_same(df_data):
    """
    比较 pandas-ta 和 talib 计算的 PSAR 结果是否相同。
    预期结果是相同，所以使用 assert_indicator_same
    """
    high_series = df_data["high"]
    low_series = df_data["low"]
    close_series = df_data["close"]

    # 定义要测试的 PSAR 参数组合
    psar_params_to_test = [(0.02, 0.02, 0.2), (0.03, 0.03, 0.3)]

    for af0_val, af_step_val, max_af_val in psar_params_to_test:
        # 使用 pandas_ta 计算 PSAR (talib=False)
        pandas_ta_psar = ta.psar(high_series,
                                 low_series,
                                 close_series,
                                 af0=af0_val,
                                 af=af_step_val,
                                 max_af=max_af_val,
                                 talib=False)

        # 使用 talib 计算 PSAR (talib=True)
        talib_psar = ta.psar(high_series,
                             low_series,
                             close_series,
                             af0=af0_val,
                             af=af_step_val,
                             max_af=max_af_val,
                             talib=True)

        psarl_key = f"PSARl_{af0_val}_{max_af_val}"
        psars_key = f"PSARs_{af0_val}_{max_af_val}"
        psaraf_key = f"PSARaf_{af0_val}_{max_af_val}"
        psarr_key = f"PSARr_{af0_val}_{max_af_val}"
        for i in [psarl_key, psars_key, psaraf_key, psarr_key]:
            # 使用 assert_indicator_different 验证两者是否相同
            assert_indicator_same(
                pandas_ta_psar[i], talib_psar[i], "psar",
                f"af0 {af0_val}, af_step {af_step_val}, max_af {max_af_val} (Pandas-TA vs TA-Lib)"
            )
