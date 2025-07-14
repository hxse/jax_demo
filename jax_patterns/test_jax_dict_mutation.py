import jax
import jax.numpy as jnp
from functools import partial


# --- 核心的 lax.scan 步骤函数 ---
# 字典就是 PyTree，无需额外注册！
@jax.jit
def _calculate_single_step_with_dict(state_dict: dict,
                                     inputs_tuple) -> tuple[dict, dict]:
    """
    模拟 lax.scan 的一个单步迭代，使用字典作为状态。
    """
    current_close, current_open, enter_long_raw, enter_short_raw = inputs_tuple
    trade_price = current_open

    # --- 重要：创建 state_dict 的副本进行修改，而不是原地修改 ---
    updated_dict = state_dict  #.copy()  # 或者 {**state_dict}

    # 模拟 _handle_entry 的逻辑
    new_pos = updated_dict['position']
    new_entry = updated_dict['entry_price']

    condition_enter_long = jnp.logical_and(enter_long_raw,
                                           updated_dict['position'] == 0.0)
    new_pos = jnp.where(condition_enter_long, 1.0, new_pos)
    new_entry = jnp.where(condition_enter_long, trade_price, new_entry)

    condition_enter_short = jnp.logical_and(enter_short_raw,
                                            updated_dict['position'] == 0.0)
    new_pos = jnp.where(condition_enter_short, -2.0, new_pos)
    new_entry = jnp.where(condition_enter_short, trade_price, new_entry)

    updated_dict['position'] = new_pos
    updated_dict['entry_price'] = new_entry
    # ... 其他字段也在这里更新 ...

    # 为了模拟 lax.scan 的输出历史，我们返回更新后的字典
    return updated_dict, updated_dict


# --- 运行示例，模拟一次完整的 lax.scan 调用流程 ---

# 初始字典状态
initial_dict_state = {
    'position': jnp.array(0.0),
    'entry_price': jnp.array(0.0),
    'exit_price': jnp.array(0.0),
    'highest_price': jnp.array(-jnp.inf),
    'lowest_price': jnp.array(jnp.inf),
    'tsl_price': jnp.array(0.0)
}

# 模拟输入数据
single_step_inputs = (
    jnp.array(101.0),  # current_close
    jnp.array(100.0),  # current_open (trade_price)
    jnp.array(True),  # enter_long_raw
    jnp.array(False)  # enter_short_raw
)

print(f"原始字典状态: {initial_dict_state}")

# 调用 JIT 编译的单步函数
new_dict_state, output_for_history = _calculate_single_step_with_dict(
    initial_dict_state, single_step_inputs)

print(f"单步更新后字典状态: {new_dict_state}")
print(f"原始字典状态是否改变? {initial_dict_state is new_dict_state}")  # 应该为 False

# 再次调用，模拟第二步，验证状态的动态传递
second_step_inputs = (jnp.array(99.0), jnp.array(98.0), jnp.array(False),
                      jnp.array(True))
final_dict_state, final_output = _calculate_single_step_with_dict(
    new_dict_state, second_step_inputs)

print(f"再次更新后字典状态: {final_dict_state}")
print(f"第一次更新字典状态是否改变? {new_dict_state is final_dict_state}")  # 应该为 False
