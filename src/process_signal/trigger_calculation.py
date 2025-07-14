from src.utils.profiling_utils import static_jit
import jax.numpy as jnp
import jax.lax as lax


@static_jit()
def _compute_tsl_price(is_long, is_short, position_prev, highest_price,
                       lowest_price, current_atr, tsl_atr_multiplier,
                       tsl_price_prev):
    """
    计算并更新追踪止损（TSL）价格。

    Args:
        is_long (bool): 是否持有多头。
        is_short (bool): 是否持有空头。
        position_prev (int): 上一K线的仓位状态。
        highest_price (float): 持多期间最高价。
        lowest_price (float): 持空期间最低价。
        current_atr (float): N-1 K线的ATR值。
        tsl_atr_multiplier (float): 追踪止损ATR倍数。
        tsl_price_prev (float): 上一K线的TSL价格。

    Returns:
        float: 更新后的TSL价格。
    """
    # 多头TSL候选价：最高价减ATR倍数
    tsl_long_candidate = highest_price - current_atr * tsl_atr_multiplier
    # 空头TSL候选价：最低价加ATR倍数
    tsl_short_candidate = lowest_price + current_atr * tsl_atr_multiplier
    # 多头时，TSL取旧TSL与候选价的较大值
    new_tsl_price = jnp.where(is_long,
                              jnp.maximum(tsl_price_prev, tsl_long_candidate),
                              tsl_price_prev)
    # 空头时，TSL取旧TSL与候选价的较小值
    new_tsl_price = jnp.where(is_short,
                              jnp.minimum(new_tsl_price, tsl_short_candidate),
                              new_tsl_price)
    # 无仓位时，TSL重置为0
    new_tsl_price = jnp.where(position_prev == 0, 0.0, new_tsl_price)
    return new_tsl_price


@static_jit()
def _check_triggers(is_long, is_short, current_close, entry_price_prev,
                    new_tsl_price, current_atr, sl_atr_multiplier,
                    tp_atr_multiplier):
    """
    检查止损（SL）、止盈（TP）和追踪止损（TSL）触发信号。

    Args:
        is_long (bool): 是否持有多头。
        is_short (bool): 是否持有空头。
        current_close (float): N-1 K线收盘价。
        entry_price_prev (float): 上一K线入场价。
        new_tsl_price (float): 当前TSL价格。
        current_atr (float): N-1 K线的ATR值。
        sl_atr_multiplier (float): 固定止损ATR倍数。
        tp_atr_multiplier (float): 止盈ATR倍数。

    Returns:
        tuple: (tsl_triggered, sl_triggered, tp_triggered)，触发信号。
    """
    # TSL触发：多头收盘价跌破TSL，或空头收盘价涨破TSL
    tsl_triggered = jnp.logical_or(
        jnp.logical_and(is_long, current_close < new_tsl_price),
        jnp.logical_and(is_short, current_close > new_tsl_price))
    # SL触发：多头收盘价跌破入场价-ATR*倍数，或空头涨破入场价+ATR*倍数
    sl_triggered = jnp.logical_or(
        jnp.logical_and(
            is_long, current_close
            < (entry_price_prev - current_atr * sl_atr_multiplier)),
        jnp.logical_and(
            is_short, current_close
            > (entry_price_prev + current_atr * sl_atr_multiplier)))
    # TP触发：多头收盘价涨过入场价+ATR*倍数，或空头跌过入场价-ATR*倍数
    tp_triggered = jnp.logical_or(
        jnp.logical_and(
            is_long, current_close
            > (entry_price_prev + current_atr * tp_atr_multiplier)),
        jnp.logical_and(
            is_short, current_close
            < (entry_price_prev - current_atr * tp_atr_multiplier)))
    return tsl_triggered, sl_triggered, tp_triggered
