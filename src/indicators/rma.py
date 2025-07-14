import jax
import jax.numpy as jnp
from jax import jit, lax
from functools import partial
from src.utils.profiling_utils import static_jit


@static_jit(static_argnames=("unroll"))
def rma_jax(series: jnp.ndarray, period: jnp.ndarray,
            unroll: int) -> jnp.ndarray:
    """
    计算给定序列的 Wilder's 移动平均线 (RMA)。

    此函数实现了 RMA 的核心逻辑，包括初始的简单移动平均 (SMA) 预热阶段
    以及随后的指数平滑。它严格遵循 JAX 的函数式编程原则，使用 `lax.scan`
    进行迭代计算以确保 JIT 兼容性。

    Args:
        series (jnp.ndarray): 输入序列 (例如，正向或负向价格差异)。
        period (jnp.ndarray): RMA 计算的周期，一个标量 JAX 数组。
        unroll (int): `lax.scan` 的展开整数，用于优化编译。

    Returns:
        jnp.ndarray: RMA 平滑后的序列。
    """
    # RMA 的初始 carry 状态: (上一个 RMA 值, 初始 SMA 的和, 初始 SMA 的计数)
    initial_rma_carry = (
        jnp.nan,  # prev_rma_value
        jnp.array(0.0, dtype=jnp.float64),  # sum for initial SMA
        jnp.array(0, dtype=jnp.int32)  # count for initial SMA
    )

    def rma_scan_body(carry, current_val_and_idx):
        """
        用于迭代计算 RMA 的 scan 主体函数。

        Args:
            carry (tuple): 从上一次迭代传递的状态:
                           (prev_rma_value, initial_sum, current_count)
            current_val_and_idx (tuple): 包含 (当前值, 当前索引) 的元组。

        Returns:
            tuple: (new_carry, current_rma_value)
                   new_carry: 下一次迭代的更新状态。
                   current_rma_value: 当前数据点的 RMA 值。
        """
        prev_rma_value, initial_sum, current_count = carry
        current_val, idx = current_val_and_idx

        is_nan_current_val = jnp.isnan(current_val)

        # 仅当当前值不是 NaN 时更新计数
        new_count = jnp.where(is_nan_current_val, current_count,
                              current_count + 1)

        # 为初始 SMA 累加
        accumulate = jnp.logical_and(new_count <= period, ~is_nan_current_val)
        new_initial_sum = jnp.where(accumulate, initial_sum + current_val,
                                    initial_sum)

        # 计算初始 SMA
        first_rma = new_initial_sum / period

        # 计算平滑后的 RMA
        smoothed_rma = (prev_rma_value * (period - 1) + current_val) / period

        # 根据计数选择正确的 RMA：预热期为 NaN，然后是 SMA，最后是 RMA 平滑值
        current_rma_value = jnp.where(
            new_count < period, jnp.nan,
            jnp.where(new_count == period, first_rma, smoothed_rma))

        # 统一处理 NaN 传播：如果当前值是 NaN 或在回溯期内，则结果为 NaN
        nan_propagation_mask = jnp.logical_or(
            is_nan_current_val, idx
            < period - 1)  # 修正为 period - 1，因为RMA的第一个有效值通常在`period`个diffs之后
        current_rma_value = jnp.where(nan_propagation_mask, jnp.nan,
                                      current_rma_value)

        # 更新下一个 prev_rma_value
        # 如果 new_count 首次达到 period，则使用 first_rma；否则，如果 new_count 仍在预热期，保持 prev_rma_value；否则使用 current_rma_value
        new_prev_rma_value = jnp.where(
            new_count == period, first_rma,
            jnp.where(new_count < period, prev_rma_value, current_rma_value))

        new_carry = (new_prev_rma_value, new_initial_sum, new_count)
        return new_carry, current_rma_value

    # 准备 scan 的输入数据：包括索引，用于处理回溯期 NaN
    # 这里的 `len(series)` 替换了原来的 `n`
    series_indexed_data = (series, jnp.arange(len(series), dtype=jnp.int32))

    # 执行 scan
    _, rma_results_scan = lax.scan(rma_scan_body,
                                   initial_rma_carry,
                                   series_indexed_data,
                                   unroll=unroll)

    return rma_results_scan


@static_jit(static_argnames=("unroll"))
def no_rma_jax(values: jnp.ndarray, period: int, unroll: int) -> jnp.ndarray:
    """
    一个“什么都不做”的 RMA 占位函数。
    它返回一个与输入 values 数组形状相同且填充了 NaN 的 JAX 数组。
    这个函数旨在与 jax.lax.cond 结合使用，当 RMA 计算被禁用时作为 false_fun。

    参数:
        values (jnp.ndarray): 输入数组（仅用于获取形状和 Dtype）。
        period (int): 周期参数（在此函数中未使用，但为匹配签名保留）。
        unroll (int): unroll 参数（在此函数中未使用，但为匹配签名保留）。

    返回:
        jnp.ndarray: 一个与 values 形状相同且填充了 NaN 的数组。
    """
    # 获取 values 数组的形状和数据类型
    output_shape = values.shape
    output_dtype = values.dtype

    # 创建一个全 NaN 的数组，与 RMA 的预期输出形状和 Dtype 匹配
    return jnp.full(output_shape, jnp.nan, dtype=output_dtype)
