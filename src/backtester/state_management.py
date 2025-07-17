from src.utils.profiling_utils import static_jit
import jax.numpy as jnp
import jax.lax as lax
from src.backtester.trade_state import TradeState, TradeInputs, TradeIntermediate, TradeOutputs, TradeStateLast
from dataclasses import replace
from typing import Tuple


@static_jit()
def _update_last(state: TradeState,
                 stateLast: TradeStateLast) -> TradeStateLast:
    return replace(stateLast, position=state.position)


@static_jit()
def _update_trade_price(intermediate: TradeIntermediate,
                        inputs: TradeInputs) -> TradeIntermediate:
    """
    初始化中间状态中的交易价格。
    """
    trade_price = inputs.open
    return replace(intermediate, trade_price=trade_price)


@static_jit()
def _update_bar_index(intermediate: TradeIntermediate) -> TradeIntermediate:
    bar_index = intermediate.bar_index + 1
    return replace(intermediate, bar_index=bar_index)


@static_jit()
def _update_entry_exit_price(
        state: TradeState,
        intermediate: TradeIntermediate) -> TradeIntermediate:
    entry_price = jnp.select(
        [
            jnp.isin(state.position, jnp.array([1, 3, -1, -3])), state.position
            == 2
        ],
        [intermediate.trade_price, state.entry_price],
        default=jnp.array(jnp.nan),
    )
    exit_price = jnp.select(
        [
            jnp.isin(state.position, jnp.array([0, 3, -3])),
        ],
        [intermediate.trade_price],
        default=jnp.array(jnp.nan),
    )
    return replace(state, entry_price=entry_price, exit_price=exit_price)


@static_jit()
def _update_position_type(state: TradeState,
                          stateLast: TradeStateLast) -> TradeState:
    """
    更新仓位状态类型：将“开仓/反手”转为“持仓”状态。
    """
    is_long = jnp.isin(state.position, jnp.array([1, 2, 3]))
    is_short = jnp.isin(state.position, jnp.array([-1, -2, -3]))
    is_long_last = jnp.isin(stateLast.position, jnp.array([1, 2, 3]))
    is_short_last = jnp.isin(stateLast.position, jnp.array([-1, -2, -3]))

    # 使用 jnp.select 来处理所有条件分支
    new_position = jnp.select([
        jnp.logical_and(is_long, is_long_last),
        jnp.logical_and(is_short, is_short_last)
    ], [jnp.array(2), jnp.array(-2)],
                              default=state.position)
    return replace(state, position=new_position)


@static_jit()
def _update_tracked_prices(state: TradeState, inputs: TradeInputs,
                           intermediate: TradeIntermediate) -> TradeState:
    """
    更新持仓期间的最高价和最低价，用于追踪止损。

    Args:
        state (TradeState): 包含当前状态的TradeState实例。
        inputs (TradeInputs): 包含当前K线收盘价的TradeInputs实例。
        intermediate (TradeIntermediate): 中间状态，包含is_long, is_short和trade_price。

    Returns:
        Tuple[TradeState, TradeIntermediate]: 更新后的状态（TradeState实例的副本）和中间状态。
    """
    # 新开多仓时，最高价为交易价格；持多时，取历史最高与收盘价的较大值
    new_highest = jnp.select(
        [jnp.isin(state.position, jnp.array([1, 3])), state.position == 2],
        [
            intermediate.trade_price,
            jnp.maximum(state.highest_price, inputs.close)
        ],
        default=jnp.array(jnp.nan),
    )

    # 新开空仓时，最低价为交易价格；持空时，取历史最低与收盘价的较小值
    new_lowest = jnp.select(
        [jnp.isin(state.position, jnp.array([-1, -3])), state.position == -2],
        [
            intermediate.trade_price,
            jnp.minimum(state.lowest_price, inputs.close)
        ],
        default=jnp.array(jnp.nan),
    )
    return replace(state, highest_price=new_highest, lowest_price=new_lowest)


@static_jit()
def _generate_trade_outputs(state: TradeState,
                            intermediate: TradeIntermediate) -> TradeOutputs:
    """
    根据交易状态和中间状态生成交易输出。
    """
    return TradeOutputs(
        positions=state.position,
        entry_prices=state.entry_price,
        exit_prices=state.exit_price,
        highest_prices=state.highest_price,
        lowest_prices=state.lowest_price,
        atr_tsl_prices=state.atr_tsl_price,
        atr_tsl_triggered_history=intermediate.atr_tsl_triggered,
        atr_sl_triggered_history=intermediate.atr_sl_triggered,
        atr_tp_triggered_history=intermediate.atr_tp_triggered)


@static_jit()
def _update_holding_bars(state: TradeState) -> TradeState:
    """
    更新持仓周期计数。

    Args:
        state (TradeState): 当前交易状态。

    Returns:
        TradeState: 更新后的交易状态。
    """
    # 根据持仓状态更新持仓周期
    holding_bars = jnp.select(
        [
            jnp.any(state.position == jnp.array([2, -2])),  # 持仓（多头或空头）
            jnp.any(
                state.position == jnp.array([1, 3, -1, -3])),  # 开多、反多、开空、反空
            state.position == 0  # 离场
        ],
        [
            state.holding_bars + 1,  # 持仓周期加1
            1,  # 设置为1
            0  # 设置为0
        ],
        default=state.holding_bars  # 默认保持不变，以防未来有新的position值
    )
    return replace(state, holding_bars=holding_bars)
