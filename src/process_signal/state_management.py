from src.utils.profiling_utils import static_jit
import jax.numpy as jnp
import jax.lax as lax


@static_jit()
def _initialize_state(position_prev, entry_price_prev, exit_price_prev,
                      highest_price_prev, lowest_price_prev, tsl_price_prev):
    """
    初始化新状态，复制上一 K 线的状态。

    Args:
        position_prev (int): 上一K线的仓位状态。
        entry_price_prev (float): 上一K线的入场价格。
        exit_price_prev (float): 上一K线的离场价格。
        highest_price_prev (float): 上一K线的持多期间最高价。
        lowest_price_prev (float): 上一K线的持空期间最低价。
        tsl_price_prev (float): 上一K线的追踪止损价格。

    Returns:
        tuple: 初始化后的状态 (new_position, new_entry_price, new_exit_price, new_highest_price, new_lowest_price, new_tsl_price)。
    """
    return position_prev, entry_price_prev, exit_price_prev, highest_price_prev, lowest_price_prev, tsl_price_prev


@static_jit()
def _reset_state(condition, new_position, new_entry_price, new_exit_price,
                 new_highest_price, new_lowest_price, new_tsl_price,
                 position_value, trade_price):
    """
    统一重置交易状态变量，用于开仓、平仓或反手操作。

    Args:
        condition (bool): 是否执行重置的条件。
        new_position (int): 当前仓位状态。
        new_entry_price (float): 当前入场价格。
        new_exit_price (float): 当前离场价格。
        new_highest_price (float): 当前持多期间最高价。
        new_lowest_price (float): 当前持空期间最低价。
        new_tsl_price (float): 当前追踪止损价格。
        position_value (int): 目标仓位值（0: 无仓，1: 开多，-1: 开空，3: 反手多，-3: 反手空）。
        trade_price (float): 当前交易价格（开盘价）。

    Returns:
        tuple: 更新后的 (new_position, new_entry_price, new_exit_price, new_highest_price, new_lowest_price, new_tsl_price)。
    """
    # 更新仓位为目标值或保持不变
    new_position = jnp.where(condition, position_value, new_position)
    # 重置入场价为交易价格（或0）
    new_entry_price = jnp.where(condition,
                                trade_price if position_value != 0 else 0.0,
                                new_entry_price)
    # 重置离场价为交易价格（或0）
    new_exit_price = jnp.where(condition,
                               trade_price if position_value == 0 else 0.0,
                               new_exit_price)
    # 重置最高价为交易价格（或负无穷）
    new_highest_price = jnp.where(
        condition, trade_price if position_value != 0 else -jnp.inf,
        new_highest_price)
    # 重置最低价为交易价格（或正无穷）
    new_lowest_price = jnp.where(
        condition, trade_price if position_value != 0 else jnp.inf,
        new_lowest_price)
    # 重置TSL价格为0
    new_tsl_price = jnp.where(condition, 0.0, new_tsl_price)
    return new_position, new_entry_price, new_exit_price, new_highest_price, new_lowest_price, new_tsl_price


@static_jit()
def _update_position_type(position_prev):
    """
    更新仓位状态类型：将“开仓/反手”转为“持仓”状态。

    Args:
        position_prev (int): 上一K线的仓位状态（0: 无仓，1: 开多，2: 持多，3: 反手多，-1: 开空，-2: 持空，-3: 反手空）。

    Returns:
        int: 更新后的仓位状态。
    """
    # 开多（1）或反手多（3）转为持多（2）
    new_position = jnp.where(position_prev == 1, 2, position_prev)
    new_position = jnp.where(position_prev == 3, 2, new_position)
    # 开空（-1）或反手空（-3）转为持空（-2）
    new_position = jnp.where(position_prev == -1, -2, new_position)
    new_position = jnp.where(position_prev == -3, -2, new_position)
    return new_position


@static_jit()
def _check_position_status(position_prev):
    """
    检查当前仓位状态，判断是否持有多头或空头。

    Args:
        position_prev (int): 上一K线的仓位状态。

    Returns:
        tuple: (is_long, is_short)，表示是否持有多头或空头。
    """
    # 判断是否持有多头（开多、持多、反手多）
    is_long = jnp.logical_or(
        jnp.logical_or(position_prev == 1, position_prev == 2),
        position_prev == 3)
    # 判断是否持有空头（开空、持空、反手空）
    is_short = jnp.logical_or(
        jnp.logical_or(position_prev == -1, position_prev == -2),
        position_prev == -3)
    return is_long, is_short


@static_jit()
def _update_tracked_prices(position_prev, is_long, is_short, trade_price,
                           current_close, highest_prev, lowest_prev):
    """
    更新持仓期间的最高价和最低价，用于追踪止损。

    Args:
        position_prev (int): 上一K线的仓位状态。
        is_long (bool): 是否持有多头。
        is_short (bool): 是否持有空头。
        trade_price (float): 当前交易价格（开盘价）。
        current_close (float): N-1 K线收盘价。
        highest_prev (float): 上一K线持多期间最高价。
        lowest_prev (float): 上一K线持空期间最低价。

    Returns:
        tuple: (new_highest, new_lowest)，更新后的最高价和最低价。
    """
    # 新开多仓时，最高价为交易价格；持多时，取历史最高与收盘价的较大值
    new_highest = jnp.where(
        jnp.logical_or(position_prev == 1, position_prev == 3), trade_price,
        jnp.where(is_long, jnp.maximum(highest_prev, current_close),
                  highest_prev))  # 这里is_long可以替换成position_prev == 2
    # 新开空仓时，最低价为交易价格；持空时，取历史最低与收盘价的较小值
    new_lowest = jnp.where(
        jnp.logical_or(position_prev == -1, position_prev == -3), trade_price,
        jnp.where(is_short, jnp.minimum(lowest_prev, current_close),
                  lowest_prev))  # is_short可以替换成position_prev == -2
    # 无仓位时，重置最高价为负无穷，最低价为正无穷
    new_highest = jnp.where(position_prev == 0, -jnp.inf, new_highest)
    new_lowest = jnp.where(position_prev == 0, jnp.inf, new_lowest)
    return new_highest, new_lowest
