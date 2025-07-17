import jax
import jax.numpy as jnp
from src.utils.profiling_utils import static_jit
from src.indicators.rma import rma_jax
from src.indicators.tr import tr_jax


@static_jit(static_argnames=("unroll"))
def atr_jax(high: jnp.ndarray, low: jnp.ndarray, close: jnp.ndarray,
            period: int, unroll: int) -> jnp.ndarray:
    """
    计算平均真实范围 (ATR)。

    参数:
        high (jnp.ndarray): 高价数组。
        low (jnp.ndarray): 低价数组。
        close (jnp.ndarray): 收盘价数组。
        period (int): ATR 的周期。

    返回:
        jnp.ndarray: ATR 值数组。
    """
    tr_values = tr_jax(high, low, close)
    atr_values = rma_jax(tr_values, period, unroll)
    return atr_values


@static_jit(static_argnames=("unroll"))
def no_atr_jax(high: jnp.ndarray, low: jnp.ndarray, close: jnp.ndarray,
               period: int, unroll: int) -> jnp.ndarray:
    """
    一个“什么都不做”的 ATR 占位函数。
    它返回一个与输入 close 数组形状相同且填充了 NaN 的 JAX 数组。
    这个函数旨在与 jax.lax.cond 结合使用，当 ATR 计算被禁用时作为 false_fun。

    参数:
        high (jnp.ndarray): 高价数组（仅用于获取形状和 Dtype）。
        low (jnp.ndarray): 低价数组（仅用于获取形状和 Dtype）。
        close (jnp.ndarray): 收盘价数组（仅用于获取形状和 Dtype）。
        period (int): 周期参数（在此函数中未使用，但为匹配签名保留）。

    返回:
        jnp.ndarray: 一个与 close 形状相同且填充了 NaN 的数组。
    """
    output_shape = close.shape
    return jnp.full(output_shape, jnp.nan)
