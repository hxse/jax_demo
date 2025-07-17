import jax
import jax.numpy as jnp
from jax import jit
from src.utils.profiling_utils import static_jit


@static_jit()
def sma_jax(close: jnp.ndarray, period: jnp.ndarray) -> jnp.ndarray:
    """
    在 JAX 中实现 SMA (Simple Moving Average) 函数，支持动态的 period。
    使用累积求和法避免了对 period 的静态要求。
    修正了前导 NaN 的数量，使其与传统 SMA 定义一致（前 period-1 个为 NaN）。

    参数:
        close (jnp.ndarray): 收盘价数组。
        period (jnp.ndarray): SMA 计算的周期。**现在可以是一个动态 JAX 数组。**

    返回:
        jnp.ndarray: SMA 结果数组。
    """
    n = len(close)

    is_nan = jnp.isnan(close)
    values_for_sum = jnp.where(is_nan, 0.0, close)  # NaN 参与求和时视为 0
    # count_for_rolling 用于计算窗口内有效数据点数量，不是为了 NaN 填充，而是为了除法
    values_for_count = jnp.where(is_nan, 0, 1)

    cumulative_sum = jnp.cumsum(values_for_sum)
    cumulative_count = jnp.cumsum(values_for_count)

    padded_cumulative_sum = jnp.concatenate([jnp.array([0.0]), cumulative_sum])
    padded_cumulative_count = jnp.concatenate(
        [jnp.array([0]), cumulative_count])

    end_indices = jnp.arange(n) + 1
    start_indices = jnp.maximum(0, jnp.arange(n) + 1 - period)  # period 是动态的

    rolling_sums = padded_cumulative_sum[end_indices] - padded_cumulative_sum[
        start_indices]

    # 关键修正点：这里我们要求窗口内的有效数据点数量必须等于 period
    # 这样才能确保只有窗口填满后才计算 SMA，否则为 NaN
    rolling_valid_count = padded_cumulative_count[
        end_indices] - padded_cumulative_count[start_indices]

    # 当 rolling_valid_count == period 且 rolling_valid_count > 0 时才计算 SMA
    # rolling_valid_count > 0 包含了处理 NaN 值的情况 (如果整个窗口都是 NaN，count为0)
    sma_result = jnp.where(
        (rolling_valid_count == period) & (rolling_valid_count > 0),
        rolling_sums / rolling_valid_count, jnp.nan)

    return sma_result


@static_jit()
def no_sma_jax(close: jnp.ndarray, period: jnp.ndarray) -> jnp.ndarray:
    """
    一个“什么都不做”的 SMA 占位函数。
    它返回一个与输入 close 数组形状相同且填充了 NaN 的 JAX 数组。
    这个函数旨在与 jax.lax.cond 结合使用，当 SMA 计算被禁用时作为 false_fun。

    参数:
        close (jnp.ndarray): 收盘价数组（仅用于获取形状和 Dtype）。
        period (jnp.ndarray): 周期参数（在此函数中未使用，但为匹配签名保留）。

    返回:
        jnp.ndarray: 一个与 close 形状相同且填充了 NaN 的数组。
    """
    # 获取 close 数组的形状和数据类型
    output_shape = close.shape
    return jnp.full(output_shape, jnp.nan)
