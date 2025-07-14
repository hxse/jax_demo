from src.utils.profiling_utils import static_jit
import jax.numpy as jnp
import jax.lax as lax
from src.process_signal.state_management import _reset_state


@static_jit()
def _combine_exit_signals(exit_long_raw, exit_short_raw, sl_triggered,
                          tp_triggered, tsl_triggered):
    """
    合并平仓信号，包括外部信号、止损、止盈和追踪止损。

    Args:
        exit_long_raw (bool): 原始平多信号。
        exit_short_raw (bool): 原始平空信号。
        sl_triggered (bool): 固定止损触发信号。
        tp_triggered (bool): 止盈触发信号。
        tsl_triggered (bool): 追踪止损触发信号。

    Returns:
        tuple: (should_exit_long, should_exit_short)，合并后的多头和空头平仓信号。
    """
    # 合并多头平仓信号：外部平仓、止损、止盈或追踪止损任一触发
    should_exit_long = jnp.logical_or(
        jnp.logical_or(exit_long_raw, sl_triggered),
        jnp.logical_or(tp_triggered, tsl_triggered))
    # 合并空头平仓信号：外部平仓、止损、止盈或追踪止损任一触发
    should_exit_short = jnp.logical_or(
        jnp.logical_or(exit_short_raw, sl_triggered),
        jnp.logical_or(tp_triggered, tsl_triggered))
    return should_exit_long, should_exit_short


@static_jit()
def _handle_exit(is_long, is_short, should_exit_long, should_exit_short,
                 trade_price, new_position, new_entry_price, new_exit_price,
                 new_highest_price, new_lowest_price, new_tsl_price):
    """
    处理平仓逻辑，重置状态变量。

    Args:
        is_long (bool): 是否持有多头。
        is_short (bool): 是否持有空头。
        should_exit_long (bool): 多头平仓信号。
        should_exit_short (bool): 空头平仓信号。
        trade_price (float): 当前交易价格（开盘价）。
        new_position (int): 当前仓位状态。
        new_entry_price (float): 当前入场价格。
        new_exit_price (float): 当前离场价格。
        new_highest_price (float): 当前持多期间最高价。
        new_lowest_price (float): 当前持空期间最低价。
        new_tsl_price (float): 当前TSL价格。

    Returns:
        tuple: 更新后的 (new_position, new_entry_price, new_exit_price, new_highest_price, new_lowest_price, new_tsl_price)。
    """
    # 平多：持多且触发平仓信号
    new_position, new_entry_price, new_exit_price, new_highest_price, new_lowest_price, new_tsl_price = _reset_state(
        jnp.logical_and(is_long, should_exit_long), new_position,
        new_entry_price, new_exit_price, new_highest_price, new_lowest_price,
        new_tsl_price, 0, trade_price)
    # 平空：持空且触发平仓信号
    new_position, new_entry_price, new_exit_price, new_highest_price, new_lowest_price, new_tsl_price = _reset_state(
        jnp.logical_and(is_short, should_exit_short), new_position,
        new_entry_price, new_exit_price, new_highest_price, new_lowest_price,
        new_tsl_price, 0, trade_price)
    return new_position, new_entry_price, new_exit_price, new_highest_price, new_lowest_price, new_tsl_price


@static_jit()
def _handle_entry(enter_long_raw, enter_short_raw, new_position, trade_price,
                  new_entry_price, new_exit_price, new_highest_price,
                  new_lowest_price, new_tsl_price):
    """
    处理普通开仓逻辑，仅在无仓位时执行。

    Args:
        enter_long_raw (bool): 开多信号。
        enter_short_raw (bool): 开空信号。
        new_position (int): 当前仓位状态。
        trade_price (float): 当前交易价格（开盘价）。
        new_entry_price (float): 当前入场价格。
        new_exit_price (float): 当前离场价格。
        new_highest_price (float): 当前持多期间最高价。
        new_lowest_price (float): 当前持空期间最低价。
        new_tsl_price (float): 当前TSL价格。

    Returns:
        tuple: 更新后的 (new_position, new_entry_price, new_exit_price, new_highest_price, new_lowest_price, new_tsl_price)。
    """
    # 开多：无仓位且有开多信号
    new_position, new_entry_price, new_exit_price, new_highest_price, new_lowest_price, new_tsl_price = _reset_state(
        jnp.logical_and(enter_long_raw, new_position == 0), new_position,
        new_entry_price, new_exit_price, new_highest_price, new_lowest_price,
        new_tsl_price, 1, trade_price)
    # 开空：无仓位且有开空信号
    new_position, new_entry_price, new_exit_price, new_highest_price, new_lowest_price, new_tsl_price = _reset_state(
        jnp.logical_and(enter_short_raw, new_position == 0), new_position,
        new_entry_price, new_exit_price, new_highest_price, new_lowest_price,
        new_tsl_price, -1, trade_price)
    return new_position, new_entry_price, new_exit_price, new_highest_price, new_lowest_price, new_tsl_price


@static_jit()
def _handle_reverse(enter_long_raw, exit_long_raw, enter_short_raw,
                    exit_short_raw, is_long, is_short, trade_price,
                    new_position, new_entry_price, new_exit_price,
                    new_highest_price, new_lowest_price, new_tsl_price):
    """
    处理反手逻辑（平仓后开反向仓位），优先级最高。

    Args:
        enter_long_raw (bool): 开多信号。
        exit_long_raw (bool): 平多信号。
        enter_short_raw (bool): 开空信号。
        exit_short_raw (bool): 平空信号。
        is_long (bool): 是否持有多头。
        is_short (bool): 是否持有免头。
        trade_price (float): 当前交易价格（开盘价）。
        new_position (int): 当前仓位状态。
        new_entry_price (float): 当前入场价格。
        new_exit_price (float): 当前离场价格。
        new_highest_price (float): 当前持多期间最高价。
        new_lowest_price (float): 当前持空期间最低价。
        new_tsl_price (float): 当前TSL价格。

    Returns:
        tuple: 更新后的 (new_position, new_entry_price, new_exit_price, new_highest_price, new_lowest_price, new_tsl_price)。
    """
    # 反手多：持空且同时有开多和平空信号
    new_position, new_entry_price, new_exit_price, new_highest_price, new_lowest_price, new_tsl_price = _reset_state(
        jnp.logical_and(is_short,
                        jnp.logical_and(enter_long_raw, exit_short_raw)),
        new_position, new_entry_price, new_exit_price, new_highest_price,
        new_lowest_price, new_tsl_price, 3, trade_price)
    # 反手空：持多且同时有开空和平多信号
    new_position, new_entry_price, new_exit_price, new_highest_price, new_lowest_price, new_tsl_price = _reset_state(
        jnp.logical_and(is_long, jnp.logical_and(enter_short_raw,
                                                 exit_long_raw)), new_position,
        new_entry_price, new_exit_price, new_highest_price, new_lowest_price,
        new_tsl_price, -3, trade_price)
    return new_position, new_entry_price, new_exit_price, new_highest_price, new_lowest_price, new_tsl_price


@static_jit()
def _preprocess_signals(enter_long, exit_long, enter_short, exit_short):
    """
    预处理交易信号，解决冲突。

    Args:
        enter_long (jax.numpy.ndarray): 原始开多信号。
        exit_long (jax.numpy.ndarray): 原始平多信号。
        enter_short (jax.numpy.ndarray): 原始开空信号。
        exit_short (jax.numpy.ndarray): 原始平空信号。

    Returns:
        tuple: 处理后的 (enter_long, exit_long, enter_short, exit_short)。
    """
    # 解决同方向开仓和平仓冲突
    conflict_long = jnp.logical_and(enter_long, exit_long)
    enter_long = jnp.logical_and(enter_long, jnp.logical_not(conflict_long))
    exit_long = jnp.logical_and(exit_long, jnp.logical_not(conflict_long))

    conflict_short = jnp.logical_and(enter_short, exit_short)
    enter_short = jnp.logical_and(enter_short, jnp.logical_not(conflict_short))
    exit_short = jnp.logical_and(exit_short, jnp.logical_not(conflict_short))

    # 解决多空开仓冲突
    conflict_enter_both = jnp.logical_and(enter_long, enter_short)
    enter_long = jnp.logical_and(enter_long,
                                 jnp.logical_not(conflict_enter_both))
    enter_short = jnp.logical_and(enter_short,
                                  jnp.logical_not(conflict_enter_both))

    # 解决多空平仓冲突
    conflict_exit_both = jnp.logical_and(exit_long, exit_short)
    exit_long = jnp.logical_and(exit_long, jnp.logical_not(conflict_exit_both))
    exit_short = jnp.logical_and(exit_short,
                                 jnp.logical_not(conflict_exit_both))

    return enter_long, exit_long, enter_short, exit_short
