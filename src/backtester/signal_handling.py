from src.utils.profiling_utils import static_jit
import jax.numpy as jnp
import jax.lax as lax
from src.backtester.trade_state import TradeState, TradeInputs, TradeIntermediate, TradeStateLast
from dataclasses import replace


@static_jit()
def _combine_exit_signals(
        inputs: TradeInputs,
        intermediate: TradeIntermediate) -> TradeIntermediate:
    """
    合并平仓信号，包括外部信号、止损、止盈和追踪止损。

    Args:
        inputs (TradeInputs): 包含原始平多和平空信号的TradeInputs实例。
        intermediate (TradeIntermediate): 包含 tp_triggered, tsl_triggered, sl_triggered 的中间交易状态。

    Returns:
        TradeIntermediate: 更新后的中间交易状态，包含 should_exit_long 和 should_exit_short。
    """
    # 合并多头平仓信号：外部平仓、止损、止盈、追踪止损或达到最大持仓周期任一触发
    should_exit_long = jnp.any(jnp.array([
        inputs.exit_long, intermediate.atr_sl_triggered,
        intermediate.atr_tp_triggered, intermediate.atr_tsl_triggered,
        intermediate.max_holding_period_triggered
    ]),
                               axis=0)
    # 合并空头平仓信号：外部平仓、止损、止盈、追踪止损或达到最大持仓周期任一触发
    should_exit_short = jnp.any(jnp.array([
        inputs.exit_short, intermediate.atr_sl_triggered,
        intermediate.atr_tp_triggered, intermediate.atr_tsl_triggered,
        intermediate.max_holding_period_triggered
    ]),
                                axis=0)
    return replace(intermediate,
                   should_exit_long=should_exit_long,
                   should_exit_short=should_exit_short)


@static_jit()
def _handle_entry(state: TradeState, stateLast: TradeStateLast,
                  inputs: TradeInputs) -> TradeState:
    """
    处理普通开仓逻辑，仅在无仓位时执行。

    Args:
        state (TradeState): 当前交易状态。
        stateLast (TradeStateLast): 上一K线周期交易状态。
        inputs (TradeInputs): 包含开多和开空信号的TradeInputs实例。
        intermediate (TradeIntermediate): 包含 min_holding_period_triggered 的中间交易状态。

    Returns:
        TradeState: 更新后的交易状态。
    """
    # 开多：无仓位且有开多信号，并且未达到最小持仓周期
    should_enter_long = jnp.all(jnp.array(
        [inputs.enter_long, stateLast.position == 0]),
                                axis=0)

    # 开空：无仓位且有开空信号，并且未达到最小持仓周期
    should_enter_short = jnp.all(jnp.array(
        [inputs.enter_short, stateLast.position == 0]),
                                 axis=0)

    # 使用 jnp.select 统一处理仓位更新逻辑
    # 优先级：先判断是否开多，再判断是否开空
    new_position = jnp.select(
        [
            should_enter_long,  # 条件1：满足开多条件
            should_enter_short  # 条件2：满足开空条件
        ],
        [
            jnp.array(1),  # 对应条件1：设为开多 (1)
            jnp.array(-1)  # 对应条件2：设为开空 (-1)
        ],
        default=state.position  # 如果以上两个条件都不满足，则保持 state.position 不变
    )

    return replace(state, position=new_position)


@static_jit()
def _handle_exit(state: TradeState, stateLast: TradeStateLast,
                 intermediate: TradeIntermediate) -> TradeState:
    """
    处理平仓逻辑，重置仓位状态。

    如果上一周期有仓位 (多头或空头) 并且本周期触发了任何平仓信号 (多头或空头平仓信号)，
    则将当前仓位重置为无仓位 (0)。否则，保持当前仓位不变。

    Args:
        state (TradeState): 当前交易状态，其 position 字段将被更新。
        stateLast (TradeStateLast): 上一K线周期的交易状态，用于判断上一周期的仓位。
        intermediate (TradeIntermediate): 包含本周期平仓信号 (should_exit_long, should_exit_short) 的中间交易状态。

    Returns:
        TradeState: 仅更新了 position 字段的 TradeState 实例副本。
    """
    # 组合多头和空头的平仓信号：只要有一个为 True，就认为应该平仓
    should_exit = jnp.logical_or(intermediate.should_exit_long,
                                 intermediate.should_exit_short)

    # 判断是否应该平仓：上一周期有仓位 (position != 0) 且本周期有平仓信号 (should_exit)
    condition_to_exit = jnp.logical_and(stateLast.position != 0, should_exit)

    # 使用 jnp.where 根据条件更新仓位
    # 如果 condition_to_exit 为 True，则将仓位设为 0 (无仓位)
    # 否则，保持 state.position 的当前值不变
    new_position = jnp.where(
        condition_to_exit,
        jnp.array(0),  # 如果条件为真，仓位设为 0
        state.position  # 如果条件为假，仓位保持不变
    )

    return replace(state, position=new_position)


@static_jit()
def _handle_reverse(state: TradeState, stateLast: TradeStateLast,
                    inputs: TradeInputs,
                    intermediate: TradeIntermediate) -> TradeState:
    """
    处理交易反手逻辑，即平掉现有仓位并立即开立反向仓位。
    此逻辑通常具有最高优先级，因为它包含了平仓和开仓两个动作。

    Args:
        state (TradeState): 当前交易状态，其 position 字段可能被更新。
        stateLast (TradeStateLast): 上一K线周期的交易状态，用于判断前一周期的仓位方向。
        inputs (TradeInputs): 包含本周期进出场信号（enter_long/short, exit_long/short）的TradeInputs实例。
        intermediate (TradeIntermediate): 中间交易状态，此函数中未直接使用其字段进行计算，但通常包含交易相关信息。

    Returns:
        TradeState: 更新后的交易状态。
                    如果满足反手多或反手空条件，position 会被设置为对应的反手状态 (3 或 -3)。
                    否则，position 保持不变。
    """
    # 判断上一周期是否为多头仓位（包括开多、持多、反手多）
    is_long_last = jnp.isin(stateLast.position, jnp.array([1, 2, 3]))
    # 判断上一周期是否为空头仓位（包括开空、持空、反手空）
    is_short_last = jnp.isin(stateLast.position, jnp.array([-1, -2, -3]))

    # 反手多条件：上一周期持有空头仓位 (is_short_last)，且本周期同时有开多信号 (inputs.enter_long) 和平空信号 (inputs.exit_short)。
    # 这表示从空头仓位转为多头仓位。
    cond_reverse_long = jnp.all(
        jnp.array([is_short_last, inputs.enter_long, inputs.exit_short]))

    # 反手空条件：上一周期持有多头仓位 (is_long_last)，且本周期同时有开空信号 (inputs.enter_short) 和平多信号 (inputs.exit_long)。
    # 这表示从多头仓位转为空头仓位。
    cond_reverse_short = jnp.all(
        jnp.array([is_long_last, inputs.enter_short, inputs.exit_long]))

    # 使用 jnp.select 根据反手条件更新仓位
    # 如果满足 cond_reverse_long，则将仓位设为 3 (反手多)
    # 如果不满足 cond_reverse_long 但满足 cond_reverse_short，则将仓位设为 -3 (反手空)
    # 否则 (两个反手条件均不满足)，仓位保持 state.position 的当前值不变。
    new_position = jnp.select(
        [cond_reverse_long, cond_reverse_short],
        [
            jnp.array(3),  # 对应条件1：设为反手多 (3)
            jnp.array(-3)  # 对应条件2：设为反手空 (-3)
        ],
        default=state.position  # 如果以上条件均不满足，则保持 state.position 不变
    )

    return replace(state, position=new_position)


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
