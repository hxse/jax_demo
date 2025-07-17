from src.utils.profiling_utils import static_jit
import jax
import jax.numpy as jnp
import jax.lax as lax
from dataclasses import replace
from typing import Tuple
from src.backtester.state_management import _update_position_type, _update_tracked_prices, _generate_trade_outputs, _update_holding_bars, _update_trade_price, _update_bar_index, _update_entry_exit_price, _update_last
from src.backtester.trigger_calculation import _compute_atr_sl, _compute_atr_tp, _compute_tsl_price, _check_triggers
from src.backtester.signal_handling import _combine_exit_signals, _handle_exit, _handle_entry, _handle_reverse
from src.backtester.trade_state import TradeState, TradeInputs, TradeOutputs, TradeIntermediate, TradeParams, TradeStateLast


@static_jit()
def _calculate_full_state_step(
    carry: Tuple[TradeState, TradeStateLast,
                 TradeIntermediate], inputs: TradeInputs, params: TradeParams
) -> Tuple[Tuple[TradeState, TradeIntermediate], TradeOutputs]:
    """
    lax.scan的核心步骤，逐根K线更新交易状态，按优先级：状态更新 -> 止损止盈 -> 平仓 -> 开仓 -> 反手。

    Args:
        carry (tuple): 上一K线的状态（TradeState, TradeIntermediate）。
        inputs (tuple): 当前K线输入（close, open, atr, enter_long, exit_long, enter_short, exit_short,
                        tsl_atr_multiplier, sl_atr_multiplier, tp_atr_multiplier）。

    Returns:
        tuple: (new_carry, output)，更新后的状态和当前K线输出。
    """
    # 解包状态和输入
    state, stateLast, intermediate = carry

    stateLast = _update_last(state, stateLast)

    # 更新 trade_price 为 open
    intermediate = _update_trade_price(intermediate, inputs)

    # 更新 bar_index +1
    intermediate = _update_bar_index(intermediate)

    # 更新 enter_price, exit_price
    state = _update_entry_exit_price(state, intermediate)

    # 更新持仓周期
    state = _update_holding_bars(state)

    # 更新追踪最高价格和最低价格 highest_price lowest_price
    state = _update_tracked_prices(state, inputs, intermediate)

    # 计算ATR止损价格
    state = _compute_atr_sl(state, inputs, params)

    # 计算ATR止盈价格
    state = _compute_atr_tp(state, inputs, params)

    # 计算TSL价格
    state = _compute_tsl_price(state, inputs, intermediate, params)

    # 检查触发信号
    intermediate = _check_triggers(state, inputs, intermediate, params)

    # 合并平仓信号
    intermediate = _combine_exit_signals(inputs, intermediate)

    # 处理普通开仓
    state = _handle_entry(state, stateLast, inputs)

    # 处理平仓 0, 不变
    state = _handle_exit(state, stateLast, intermediate)

    # 处理反手 3, -3, 不变
    state = _handle_reverse(state, stateLast, inputs, intermediate)

    # # 更新仓位类型, 2, -2, 不变, 需要获取last状态
    state = _update_position_type(state, stateLast)

    # 生成交易输出
    output = _generate_trade_outputs(state, intermediate)
    return (state, stateLast, intermediate), output
