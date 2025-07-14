import jax
import jax.numpy as jnp
from src.utils.profiling_utils import static_jit

# PSAR 参数默认值
PSAR_AF0 = 0.02
PSAR_AF_STEP = 0.02
PSAR_MAX_AF = 0.20


@static_jit(static_argnames=("unroll"))
def psar_jax(
    high: jnp.ndarray,
    low: jnp.ndarray,
    close: jnp.ndarray,
    af0: float = PSAR_AF0,
    af_step: float = PSAR_AF_STEP,
    max_af: float = PSAR_MAX_AF,
    unroll: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """计算抛物线转向指标 (Parabolic SAR)，模仿 Pandas TA 逻辑。"""
    n = len(close)

    if n < 2:
        nan_array = jnp.full(close.shape, jnp.nan, dtype=close.dtype)
        return nan_array, nan_array, nan_array, nan_array

    # 初始趋势判断
    dn_diff = low[0] - low[1]
    up_diff = high[1] - high[0]
    is_falling_initial = jnp.logical_and(dn_diff > up_diff, dn_diff > 0)

    # 初始化 SAR, EP, AF 和趋势
    initial_sar = close[0]
    initial_ep = jax.lax.cond(is_falling_initial, lambda: low[0],
                              lambda: high[0])
    initial_af = af0
    initial_is_long = ~is_falling_initial

    # scan 的初始状态
    initial_state = (
        initial_is_long,
        initial_sar,
        initial_ep,
        initial_af,
        high[0],
        low[0],
    )

    def _psar_scan_body(state, inputs):
        """jax.lax.scan 的循环体，精确模拟 pandas-ta 的状态更新顺序。"""
        is_long, prev_sar, prev_ep, prev_af, prev_high, prev_low = state
        current_high, current_low = inputs

        # 1. 计算非反转情况下的 ep 和 af
        ep_update_cond = jax.lax.cond(is_long, lambda: current_high > prev_ep,
                                      lambda: current_low < prev_ep)
        ep_no_rev = jax.lax.cond(
            ep_update_cond,
            lambda: jax.lax.cond(is_long, lambda: jnp.maximum(
                prev_ep, current_high), lambda: jnp.minimum(
                    prev_ep, current_low)), lambda: prev_ep)
        af_no_rev = jax.lax.cond(
            ep_update_cond, lambda: jnp.minimum(max_af, prev_af + af_step),
            lambda: prev_af)

        # 2. 计算候选 sar
        sar_candidate = prev_sar + prev_af * (prev_ep - prev_sar)

        # 3. 【关键修正】使用'候选'sar判断是否反转，而不是'调整后'的sar
        reversal_occurred = jax.lax.cond(is_long,
                                         lambda: current_low < sar_candidate,
                                         lambda: current_high > sar_candidate)

        # 4. 调整 sar
        sar_adjusted = jax.lax.cond(
            is_long, lambda: jnp.minimum(sar_candidate, prev_low),
            lambda: jnp.maximum(sar_candidate, prev_high))

        # 5. 根据是否反转，计算最终的 sar, ep, af, is_long 作为下一步的状态
        final_sar = jax.lax.cond(reversal_occurred, lambda: ep_no_rev,
                                 lambda: sar_adjusted)
        final_is_long = jax.lax.cond(reversal_occurred, lambda: ~is_long,
                                     lambda: is_long)
        final_af = jax.lax.cond(reversal_occurred, lambda: af0,
                                lambda: af_no_rev)
        final_ep = jax.lax.cond(
            reversal_occurred, lambda: jax.lax.cond(
                final_is_long, lambda: current_high, lambda: current_low),
            lambda: ep_no_rev)

        # 准备下一次迭代的状态和本次迭代的输出
        next_state = (final_is_long, final_sar, final_ep, final_af,
                      current_high, current_low)
        output = (final_sar, final_is_long, final_af,
                  reversal_occurred.astype(close.dtype))
        return next_state, output

    # 执行 scan
    _, (sar_scan, is_long_scan, af_scan, reversal_scan) = jax.lax.scan(
        _psar_scan_body,
        initial_state,
        (high[1:], low[1:]),
        unroll=unroll,
    )

    # ---- 构建最终输出 ----

    # PSAR Long / Short
    final_psar_long = jnp.full_like(close, jnp.nan)
    final_psar_short = jnp.full_like(close, jnp.nan)

    long_values = jnp.where(is_long_scan, sar_scan, jnp.nan)
    short_values = jnp.where(~is_long_scan, sar_scan, jnp.nan)

    final_psar_long = final_psar_long.at[1:].set(long_values)
    final_psar_short = final_psar_short.at[1:].set(short_values)

    # AF 加速因子
    final_psar_af = jnp.full_like(
        close, jnp.nan).at[0].set(initial_af).at[1:].set(af_scan)

    # Reversal 反转信号
    final_psar_reversal = jnp.full_like(close, 0.0).at[1:].set(reversal_scan)

    return final_psar_long, final_psar_short, final_psar_af, final_psar_reversal


@static_jit(static_argnames=("unroll"))
def no_psar_jax(
    high: jnp.ndarray,
    low: jnp.ndarray,
    close: jnp.ndarray,
    af0: float = PSAR_AF0,
    af_step: float = PSAR_AF_STEP,
    max_af: float = PSAR_MAX_AF,
    unroll: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """PSAR 占位函数，返回全 NaN 数组。"""
    output_shape = close.shape
    nan_array = jnp.full(output_shape, jnp.nan, dtype=close.dtype)
    return nan_array, nan_array, nan_array, nan_array
