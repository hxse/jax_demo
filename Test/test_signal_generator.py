import jax.numpy as jnp
import numpy as np
import pytest

from src.jax_core.signal_generator import generate_signal
from src.utils.config_utils import indicator_params_template
from src.signals.signal_template import sma_simple
from src.examples.run_benchmark import run_benchmark
from Test.test_utils import assert_indicator_same
from src.signals.signal_processors import get_prev_arr


def test_signal_generator_sma_simple(df_data, np_data):
    """
    测试信号生成器在 SMA 简单策略下的行为。

    验证 `run_benchmark` 生成的交易信号与手动计算的预期信号是否一致，
    特别关注信号的后处理逻辑（避免连续信号）。

    参数:
        df_data: 用于测试的 DataFrame 格式数据。
        np_data: 用于测试的 NumPy 数组格式数据。
    """
    # 1. 准备 OHLCV 数据
    time = np_data[:, 0]
    open = np_data[:, 1]
    high = np_data[:, 2]
    low = np_data[:, 3]
    close = np_data[:, 4]
    volume = np_data[:, 5]

    # 2. 配置指标参数，启用 SMA 和 SMA2，并对 "sma" 信号进行入场/出场后处理。
    indicator_params = indicator_params_template(period=[14, 15, 1],
                                                 enable_list=["sma", "sma2"],
                                                 enable_enter_prev=["sma"],
                                                 enable_exit_prev=["sma"])

    # 3. 运行 benchmark 获取指标和信号结果
    result = run_benchmark(np_data,
                           indicator_params=indicator_params,
                           enable_cpu=True,
                           enable_gpu=False,
                           enable_run_second=False)

    # 4. 从 benchmark 结果中提取 SMA 指标计算值
    cpu_result = result["cpu_result"]
    indicators = cpu_result["micro"]["indicators"]
    signal = cpu_result["micro"]["signal"]
    sma_cpu_result = indicators["sma"]["sma_result"][0]
    sma2_cpu_result = indicators["sma2"]["sma2_result"][0]

    # 5. 手动计算原始预期信号 (基于 SMA 交叉策略)
    expected_enter_long_raw = (sma_cpu_result > sma2_cpu_result)
    expected_exit_long_raw = (sma_cpu_result < sma2_cpu_result)
    expected_enter_short_raw = (sma_cpu_result < sma2_cpu_result)
    expected_exit_short_raw = (sma_cpu_result > sma2_cpu_result)

    # 6. 模拟信号后处理
    # `run_benchmark` 会自动应用 `enable_enter_prev` 和 `enable_exit_prev` 逻辑，
    # 即避免连续信号。为了精确比较，这里手动模拟相同的后处理。
    expected_enter_long = expected_enter_long_raw & ~get_prev_arr(
        expected_enter_long_raw)
    expected_exit_long = expected_exit_long_raw & ~get_prev_arr(
        expected_exit_long_raw)
    expected_enter_short = expected_enter_short_raw & ~get_prev_arr(
        expected_enter_short_raw)
    expected_exit_short = expected_exit_short_raw & ~get_prev_arr(
        expected_exit_short_raw)

    # 7. 从 run_benchmark 结果中提取生成的信号 (JAX 数组)
    generated_enter_long = signal["enter_long"][0]
    generated_exit_long = signal["exit_long"][0]
    generated_enter_short = signal["enter_short"][0]
    generated_exit_short = signal["exit_short"][0]

    # 8. 比较预期结果与实际生成结果 (转换为 NumPy 数组进行比较)
    assert_indicator_same(np.asarray(expected_enter_long),
                          np.asarray(generated_enter_long), "enter_long",
                          "sma_simple")
    assert_indicator_same(np.asarray(expected_exit_long),
                          np.asarray(generated_exit_long), "exit_long",
                          "sma_simple")
    assert_indicator_same(np.asarray(expected_enter_short),
                          np.asarray(generated_enter_short), "enter_short",
                          "sma_simple")
    assert_indicator_same(np.asarray(expected_exit_short),
                          np.asarray(generated_exit_short), "exit_short",
                          "sma_simple")
