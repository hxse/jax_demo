from dataclasses import replace
from src.backtester.trade_state import TradeState, TradeInputs, TradeIntermediate, TradeParams
from src.utils.profiling_utils import static_jit
import jax.numpy as jnp
import jax.lax as lax


@static_jit()
def _compute_atr_sl(state: TradeState, inputs: TradeInputs,
                    params: TradeParams) -> TradeState:
    """
    计算ATR止损价格。
    """
    atr_sl_price = state.entry_price - inputs.raw_atr_values * params.atr_sl_multiplier
    return replace(state, atr_sl_price=atr_sl_price)


@static_jit()
def _compute_atr_tp(state: TradeState, inputs: TradeInputs,
                    params: TradeParams) -> TradeState:
    """
    计算ATR止盈价格。
    """
    atr_tp_price = state.entry_price + inputs.raw_atr_values * params.atr_tp_multiplier
    return replace(state, atr_tp_price=atr_tp_price)


@static_jit()
def _compute_tsl_price(state: TradeState, inputs: TradeInputs,
                       intermediate: TradeIntermediate,
                       params: TradeParams) -> TradeState:
    """
    计算并更新追踪止损（TSL）价格。
    """
    is_long = jnp.isin(state.position, jnp.array([1, 2, 3]))
    is_short = jnp.isin(state.position, jnp.array([-1, -2, -3]))

    tsl_long_candidate = state.highest_price - inputs.raw_atr_values * params.atr_tsl_multiplier
    tsl_short_candidate = state.lowest_price + inputs.raw_atr_values * params.atr_tsl_multiplier

    # 使用 jnp.select 同时处理多头、空头和无仓位情况
    new_atr_tsl_price = jnp.select(
        [
            is_long,  # 条件1：多头持仓
            is_short  # 条件2：空头持仓
        ],
        [
            jnp.maximum(state.atr_tsl_price, tsl_long_candidate),  # 对应多头时
            jnp.minimum(state.atr_tsl_price, tsl_short_candidate)  # 对应空头时
        ],
        default=jnp.array(jnp.nan)  # 如果既不是多头也不是空头（例如 position == 0），则设为 NaN
    )

    return replace(state, atr_tsl_price=new_atr_tsl_price)


@static_jit()
def _check_triggers(state: TradeState, inputs: TradeInputs,
                    intermediate: TradeIntermediate,
                    params: TradeParams) -> TradeIntermediate:
    """
    检查止损（SL）、止盈（TP）和追踪止损（TSL）触发信号。

    Args:
        state (TradeState): 包含当前状态的TradeState实例。
        inputs (TradeInputs): 包含当前K线收盘价、ATR值、止损止盈ATR倍数的TradeInputs实例。
        intermediate (TradeIntermediate): 包含交易中间状态的对象。

    Returns:
        TradeIntermediate: 更新后的交易中间状态。
    """
    is_long = jnp.isin(state.position, jnp.array([1, 2, 3]))
    is_short = jnp.isin(state.position, jnp.array([-1, -2, -3]))
    # TSL触发：多头收盘价跌破TSL，或空头收盘价涨破TSL
    atr_tsl_triggered = jnp.logical_or(
        jnp.logical_and(is_long, inputs.close < state.atr_tsl_price),
        jnp.logical_and(is_short, inputs.close > state.atr_tsl_price))
    # SL触发：多头收盘价跌破入场价-ATR*倍数，或空头涨破入场价+ATR*倍数
    atr_sl_triggered = jnp.logical_or(
        jnp.logical_and(
            is_long, inputs.close
            < (state.entry_price -
               inputs.raw_atr_values * params.atr_sl_multiplier)),
        jnp.logical_and(
            is_short, inputs.close
            > (state.entry_price +
               inputs.raw_atr_values * params.atr_sl_multiplier)))
    # TP触发：多头收盘价涨过入场价+ATR*倍数，或空头跌过入场价-ATR*倍数
    atr_tp_triggered = jnp.logical_or(
        jnp.logical_and(
            is_long, inputs.close
            > (state.entry_price +
               inputs.raw_atr_values * params.atr_tp_multiplier)),
        jnp.logical_and(
            is_short, inputs.close
            < (state.entry_price -
               inputs.raw_atr_values * params.atr_tp_multiplier)))
    # 检查是否达到最小持仓周期
    min_holding_period_triggered = jnp.logical_and(
        jnp.logical_or(is_long, is_short), state.holding_bars
        >= params.min_holding_period)

    # 检查是否达到最大持仓周期
    max_holding_period_triggered = jnp.logical_and(
        jnp.logical_or(is_long, is_short), state.holding_bars
        >= params.max_holding_period)

    return replace(intermediate,
                   atr_tsl_triggered=atr_tsl_triggered,
                   atr_sl_triggered=atr_sl_triggered,
                   atr_tp_triggered=atr_tp_triggered,
                   min_holding_period_triggered=min_holding_period_triggered,
                   max_holding_period_triggered=max_holding_period_triggered)
