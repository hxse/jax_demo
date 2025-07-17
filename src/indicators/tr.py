import jax
import jax.numpy as jnp
from src.utils.profiling_utils import static_jit


@static_jit()
def tr_jax(high: jnp.ndarray, low: jnp.ndarray,
           close: jnp.ndarray) -> jnp.ndarray:
    """
    计算真实波动范围 (True Range, TR)。

    参数:
        high (jnp.ndarray): 高价数组。
        low (jnp.ndarray): 低价数组。
        close (jnp.ndarray): 收盘价数组。

    返回:
        jnp.ndarray: 真实波动范围数组，第一个元素为 NaN。
    """
    range1 = high - low
    range2 = jnp.abs(high - jnp.roll(close, 1))
    range3 = jnp.abs(low - jnp.roll(close, 1))
    true_range = jnp.max(jnp.array([range1, range2, range3]), axis=0)
    true_range = true_range.at[0].set(jnp.nan)
    return true_range


@static_jit()
def no_tr_jax(high: jnp.ndarray, low: jnp.ndarray,
              close: jnp.ndarray) -> jnp.ndarray:
    """
    一个“什么都不做”的 TR 占位函数。
    它返回一个与输入 close 数组形状相同且填充了 NaN 的 JAX 数组。
    这个函数旨在与 jax.lax.cond 结合使用，当 TR 计算被禁用时作为 false_fun。

    参数:
        high (jnp.ndarray): 高价数组（仅用于获取形状和 Dtype）。
        low (jnp.ndarray): 低价数组（仅用于获取形状和 Dtype）。
        close (jnp.ndarray): 收盘价数组（仅用于获取形状和 Dtype）。

    返回:
        jnp.ndarray: 一个与 close 形状相同且填充了 NaN 的数组。
    """
    output_shape = close.shape
    return jnp.full(output_shape, jnp.nan)
