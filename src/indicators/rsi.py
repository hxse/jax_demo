import jax
import jax.numpy as jnp
from jax import jit, lax
from functools import partial
from src.utils.profiling_utils import static_jit
from src.indicators.rma import rma_jax

from src.utils.constants import GLOBAL_EPSILON


@static_jit(static_argnames=("unroll"))
def rsi_jax(close: jnp.ndarray, period: jnp.ndarray,
            unroll: int) -> jnp.ndarray:
    """
    计算给定收盘价序列的相对强度指数 (RSI)。

    此函数实现了 RSI 计算逻辑，紧密模仿 TA-Lib 的行为，包括初始的 SMA 类似平均、
    随后的 Wilder 移动平均 (RMA) 平滑，以及精确的 NaN 值传播。它严格遵守 JAX 的
    函数式编程原则，使用 `lax.cond` 进行控制流，并利用外部 `rma_jax` 函数进行
    迭代计算以确保 JIT 兼容性。

    Args:
        close (jnp.ndarray): 收盘价的 1D JAX 数组 (推荐 float64 以获得精度)。
        period (jnp.ndarray): 代表 RSI 周期的标量 JAX 数组 (推荐 int32)。
        unroll (int): `lax.scan` 的展开整数，用于优化编译。

    Returns:
        jnp.ndarray: 包含计算出的 RSI 值的 1D JAX 数组 (float64)。
                     前导 NaN 值和无效输入的 NaN 传播与 TA-Lib 保持一致。
    """
    n = len(close)

    def _calculate_rsi_for_period_greater_than_one(close_arr, period_val):
        """
        内部函数，计算周期大于 1 时的 RSI。
        利用 `rma_jax` 函数进行移动平均计算。
        """
        # 计算价格差异，为与 TA-Lib 行为一致，第一个差异设为 NaN
        diff = jnp.diff(close_arr, prepend=jnp.array([jnp.nan]))

        # 分离正向和负向差异
        positive_diff = jnp.maximum(0.0, diff)
        negative_diff = jnp.maximum(0.0, -diff)

        # 使用 rma_jax 计算平均增益和损失
        avg_gain = rma_jax(positive_diff, period_val, unroll)
        avg_loss = rma_jax(negative_diff, period_val, unroll)

        # 计算 RS 和 RSI
        total_avg_change = avg_gain + avg_loss

        # 处理 avg_loss 接近零的情况，避免除以零，将 RS 设为无穷大
        rs_val = jnp.where(
            jnp.abs(avg_loss) < GLOBAL_EPSILON, jnp.inf, avg_gain / avg_loss)

        rsi_val_from_formula = 100.0 - (100.0 / (1.0 + rs_val))

        # 确定最终 RSI 值：
        # 1. 如果 avg_gain 或 avg_loss 为 NaN，则 RSI 为 NaN。
        # 2. 如果当前索引在预热期内 (< period_val)，则 RSI 为 NaN。
        # 3. 如果 total_avg_change 接近零（即 avg_gain 和 avg_loss 都很小或为零），RSI 为 0.0。
        # 4. 否则，使用公式计算的 RSI 值。

        # 定义 NaN 和 0.0 的条件
        is_nan_avg = jnp.logical_or(jnp.isnan(avg_gain), jnp.isnan(avg_loss))
        is_warmup_period = jnp.arange(n) < period_val
        is_zero_avg_change = jnp.logical_and(
            jnp.arange(n) >= period_val,
            jnp.abs(total_avg_change) < GLOBAL_EPSILON)

        final_rsi = jnp.where(
            is_nan_avg,
            jnp.nan,  # 首先处理 NaN 情况
            jnp.where(
                is_warmup_period,
                jnp.nan,  # 然后处理预热期 NaN
                jnp.where(
                    is_zero_avg_change,
                    0.0,  # 接着处理总平均变化为零的情况
                    rsi_val_from_formula  # 最后是正常计算值
                )))

        return final_rsi

    # 使用 lax.cond 进行条件逻辑，处理 period=1 和 period > 1 的情况
    return lax.cond(
        period == 1, lambda close_arr_val, period_val_dummy: jnp.full_like(
            close_arr_val, jnp.nan),
        _calculate_rsi_for_period_greater_than_one, close, period)


@static_jit(static_argnames=("unroll"))  # 保持与 rsi_jax 相同的装饰器，以确保签名一致性
def no_rsi_jax(close: jnp.ndarray, period: int, unroll: int) -> jnp.ndarray:
    """
    一个“什么都不做”的 RSI 占位函数。
    它返回一个与输入 close 数组形状相同且填充了 NaN 的 JAX 数组。
    这个函数旨在与 jax.lax.cond 结合使用，当 RSI 计算被禁用时作为 false_fun。

    参数:
        close (jnp.ndarray): 收盘价数组（仅用于获取形状和 Dtype）。
        period (int): 周期参数（在此函数中未使用，但为匹配签名保留）。
        unroll (int): unroll 参数（在此函数中未使用，但为匹配签名保留）。

    返回:
        jnp.ndarray: 一个与 close 形状相同且填充了 NaN 的数组。
    """
    output_shape = close.shape
    return jnp.full(output_shape, jnp.nan)
