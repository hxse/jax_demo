from src.utils.profiling_utils import static_jit
import jax.numpy as jnp
import jax.lax as lax
from src.process_signal.state_management import _initialize_state, _update_position_type, _check_position_status, _update_tracked_prices
from src.process_signal.trigger_calculation import _compute_tsl_price, _check_triggers
from src.process_signal.signal_handling import _combine_exit_signals, _handle_exit, _handle_entry, _handle_reverse


@static_jit()
def _calculate_full_state_step(state, inputs):
    """
    lax.scan的核心步骤，逐根K线更新交易状态，按优先级：状态更新 -> 止损止盈 -> 平仓 -> 开仓 -> 反手。

    Args:
        state (tuple): 上一K线的状态（position, entry_price, exit_price, highest_price, lowest_price, tsl_price）。
        inputs (tuple): 当前K线输入（close, open, atr, enter_long, exit_long, enter_short, exit_short,
                        tsl_atr_multiplier, sl_atr_multiplier, tp_atr_multiplier）。

    Returns:
        tuple: (new_state, output)，更新后的状态和当前K线输出。
    """
    # 解包状态和输入
    position_prev, entry_price_prev, exit_price_prev, highest_price_prev, lowest_price_prev, tsl_price_prev = state
    current_close, current_open, current_atr, enter_long_raw, exit_long_raw, \
    enter_short_raw, exit_short_raw, tsl_atr_multiplier, sl_atr_multiplier, tp_atr_multiplier = inputs

    # 交易价格为当前K线开盘价
    trade_price = current_open

    # 初始化新状态
    new_position, new_entry_price, new_exit_price, new_highest_price, new_lowest_price, new_tsl_price = _initialize_state(
        position_prev, entry_price_prev, exit_price_prev, highest_price_prev,
        lowest_price_prev, tsl_price_prev)

    # 更新仓位类型
    new_position = _update_position_type(position_prev)

    # 检查持仓状态
    is_long, is_short = _check_position_status(position_prev)

    # 更新追踪价格
    new_highest_price, new_lowest_price = _update_tracked_prices(
        position_prev, is_long, is_short, trade_price, current_close,
        highest_price_prev, lowest_price_prev)

    # 计算TSL价格
    new_tsl_price = _compute_tsl_price(is_long, is_short, position_prev,
                                       new_highest_price, new_lowest_price,
                                       current_atr, tsl_atr_multiplier,
                                       tsl_price_prev)

    # 检查触发信号
    tsl_triggered, sl_triggered, tp_triggered = _check_triggers(
        is_long, is_short, current_close, entry_price_prev, new_tsl_price,
        current_atr, sl_atr_multiplier, tp_atr_multiplier)

    # 合并平仓信号
    should_exit_long, should_exit_short = _combine_exit_signals(
        exit_long_raw, exit_short_raw, sl_triggered, tp_triggered,
        tsl_triggered)

    # 处理普通开仓
    new_position, new_entry_price, new_exit_price, new_highest_price, new_lowest_price, new_tsl_price = _handle_entry(
        enter_long_raw, enter_short_raw, new_position, trade_price,
        new_entry_price, new_exit_price, new_highest_price, new_lowest_price,
        new_tsl_price)

    # 处理平仓
    new_position, new_entry_price, new_exit_price, new_highest_price, new_lowest_price, new_tsl_price = _handle_exit(
        is_long, is_short, should_exit_long, should_exit_short, trade_price,
        new_position, new_entry_price, new_exit_price, new_highest_price,
        new_lowest_price, new_tsl_price)

    # 处理反手
    new_position, new_entry_price, new_exit_price, new_highest_price, new_lowest_price, new_tsl_price = _handle_reverse(
        enter_long_raw, exit_long_raw, enter_short_raw, exit_short_raw,
        is_long, is_short, trade_price, new_position, new_entry_price,
        new_exit_price, new_highest_price, new_lowest_price, new_tsl_price)

    # 返回新状态和输出
    new_state = (new_position, new_entry_price, new_exit_price,
                 new_highest_price, new_lowest_price, new_tsl_price)
    output = (new_position, new_entry_price, new_exit_price, new_highest_price,
              new_lowest_price, new_tsl_price, tsl_triggered, sl_triggered,
              tp_triggered)
    return new_state, output
