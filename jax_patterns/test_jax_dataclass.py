import jax
import jax.numpy as jnp
from dataclasses import dataclass, field, replace  # 导入 field
from jax.tree_util import register_dataclass  # 导入这个更方便的装饰器


# --- 1. 定义并注册 TradingState dataclass 为 PyTree (简洁版) ---
@register_dataclass  # <--- 使用这个装饰器！
@dataclass
class TradingState:
    position: jnp.ndarray
    entry_price: jnp.ndarray
    exit_price: jnp.ndarray
    highest_price: jnp.ndarray
    lowest_price: jnp.ndarray
    tsl_price: jnp.ndarray
    # 如果未来需要添加静态配置，可以这样添加：
    # config_id: str = field(default="default_config", metadata=dict(static=True))


# --- 2. 你的工具函数 (保持不变，没有 @jax.jit) ---
def _handle_entry(state: TradingState, enter_long_raw: jnp.ndarray,
                  enter_short_raw: jnp.ndarray,
                  trade_price: jnp.ndarray) -> TradingState:
    """处理开仓逻辑。"""
    new_pos = state.position
    new_entry = state.entry_price

    condition_enter_long = jnp.logical_and(enter_long_raw,
                                           state.position == 0.0)
    new_pos = jnp.where(condition_enter_long, 1.0, new_pos)
    new_entry = jnp.where(condition_enter_long, trade_price, new_entry)

    condition_enter_short = jnp.logical_and(enter_short_raw,
                                            state.position == 0.0)
    new_pos = jnp.where(condition_enter_short, -2.0, new_pos)
    new_entry = jnp.where(condition_enter_short, trade_price, new_entry)

    return replace(state, position=new_pos, entry_price=new_entry)


# --- 3. 核心的 lax.scan 步骤函数（现在使用 jax.jit） ---
@jax.jit
def _calculate_single_step(state: TradingState,
                           inputs_tuple) -> tuple[TradingState, TradingState]:
    """
    模拟 lax.scan 的一个单步迭代。
    """
    current_close, current_open, enter_long_raw, enter_short_raw = inputs_tuple
    trade_price = current_open

    updated_state = _handle_entry(state, enter_long_raw, enter_short_raw,
                                  trade_price)
    return updated_state, updated_state


# --- 4. 运行示例 ---
initial_state = TradingState(position=jnp.array(0.0),
                             entry_price=jnp.array(0.0),
                             exit_price=jnp.array(0.0),
                             highest_price=jnp.array(-jnp.inf),
                             lowest_price=jnp.array(jnp.inf),
                             tsl_price=jnp.array(0.0))

single_step_inputs = (jnp.array(101.0), jnp.array(100.0), jnp.array(True),
                      jnp.array(False))

print(f"原始状态: {initial_state}")
new_state, _ = _calculate_single_step(initial_state, single_step_inputs)
print(f"单步更新后状态: {new_state}")
print(f"原始状态是否改变? {initial_state is new_state}")

second_step_inputs = (jnp.array(99.0), jnp.array(98.0), jnp.array(False),
                      jnp.array(True))
final_state, _ = _calculate_single_step(new_state, second_step_inputs)
print(f"再次更新后状态: {final_state}")
print(f"第一次更新状态是否改变? {new_state is final_state}")
