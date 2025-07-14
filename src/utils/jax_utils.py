import numpy as np
import jax.numpy as jnp
import jax  # 导入 jax 以识别 jax.Array
import collections  # 导入 collections 模块


def block_jax_output(output):
    """
    递归地阻塞 JAX 函数的输出。
    能够处理任意嵌套的字典、列表、元组等结构，并阻塞其中所有的 JAX 数组。

    Args:
        output: JAX 函数的返回值，可以是单个 JAX 数组，
                或包含 JAX 数组及其他结构（字典、列表、元组等）的任意嵌套结构。
    """
    if hasattr(output, 'block_until_ready') and callable(
            output.block_until_ready):
        # 如果是 JAX 数组，直接阻塞
        output.block_until_ready()
    elif isinstance(output, dict):
        # 如果是字典，遍历值并递归调用自身
        for value in output.values():
            block_jax_output(value)
    elif isinstance(output, (list, tuple)):
        # 如果是列表或元组，遍历元素并递归调用自身
        for item in output:
            block_jax_output(item)
    # 对于其他类型（如 int, float, str, np.ndarray 等），则不执行任何操作


import numpy as np
# 假设你已经安装了 JAX，如果没安装，可以暂时注释掉 JAX 相关的部分
# import jax
# import jax.numpy as jnp

import numpy as np
# 假设你已经安装了 JAX，如果没安装，可以暂时注释掉 JAX 相关的部分
import jax
import jax.numpy as jnp


def get_adapted_indicator_functions(indicator_info: dict, setting: int):
    """
    根据指标信息和静态的unroll值，返回适配JAX lax.cond的func和no_func。

    Args:
        indicator_info (dict): 来自 indicator_functions 字典的单个指标信息，
                               包含 'func', 'no_func' 和可选的 'unroll' 标记。
        setting (object): 一个包含 `unroll` 属性的对象，`setting.unroll` 是一个静态整数值，用于控制unroll行为。

    Returns:
        tuple: 包含 (adapted_func, adapted_no_func)。
    """
    ori_func = indicator_info["func"]
    ori_no_func = indicator_info["no_func"]

    if "unroll" in indicator_info and indicator_info["unroll"]:
        # 如果需要unroll，则将静态的unroll值通过闭包传递
        func = lambda params: ori_func(params[0], params[1], setting.unroll)
        no_func = lambda params: ori_no_func(params[0], params[1], setting.
                                             unroll)
    else:
        # 否则，只传递params解包后的动态参数
        func = lambda params: ori_func(params[0], params[1])
        no_func = lambda params: ori_no_func(params[0], params[1])

    return func, no_func


def reset_to_template(data_structure):
    """
    递归地遍历嵌套字典，根据键和值类型生成模板。

    对于字典中的每个键值对 (k, v):
    - 如果 v 是一个非空字典，则递归处理 v。
    - 如果 v 是一个空字典，则将其替换为 None。
    - 如果 v 不是字典 (即为叶子节点)，则根据键 k 的规则来赋值：
      - 如果 k 以下划线开头，该叶子节点的值替换为 None。
      - 否则 (k 不以下划线开头)，该叶子节点的值替换为 0。

    如果传入的 data_structure 不是字典类型，则直接返回 None。

    Args:
        data_structure (any): 任意嵌套的数据结构，预期顶层是字典。

    Returns:
        any: 处理后的新字典结构，或 None（如果输入不是字典或空字典）。
    """
    if isinstance(data_structure, dict):
        # 新增的检查：如果当前字典为空，直接返回 None
        if not data_structure:  # 等同于 if len(data_structure) == 0:
            return None

        new_dict = {}
        for k, v in data_structure.items():
            if isinstance(v, dict):
                # 如果值 v 是一个字典（非空字典会进入这里），则递归处理它
                # 递归调用会处理嵌套的空字典，并将其转换为 None
                new_dict[k] = reset_to_template(v)
            else:
                # 如果值 v 不是字典（即为叶子节点）
                if k.startswith('_'):
                    new_dict[k] = None
                else:
                    new_dict[k] = 0
        return new_dict
    else:
        # 如果传入的 data_structure 本身不是字典，也返回 None
        return None


def strip_key_prefix(data_structure):
    """
    递归地遍历嵌套字典，将所有以 '_' 开头的键重命名为不带 '_' 的同名键，
    并将其值复制过去。原始的带 '_' 的键将被删除。
    此版本尝试保持原始键的顺序。

    Args:
        data_structure (any): 任意嵌套的数据结构，预期顶层是字典。

    Returns:
        any: 处理后的新字典结构，如果输入不是字典则返回原始数据。
             返回的字典将是 collections.OrderedDict 或标准的 dict (在 Python 3.7+ 中默认有序)。
    """
    if not isinstance(data_structure, dict):
        return data_structure

    # 使用 OrderedDict 来确保在 Python 3.6 及更早版本中也保持顺序
    # 在 Python 3.7+ 中，标准 dict 默认保持插入顺序，但使用 OrderedDict 更显式
    new_dict = collections.OrderedDict()

    for k, v in data_structure.items():
        processed_key = k
        if k.startswith('_'):
            processed_key = k[1:]  # 移除前导下划线

        # 递归处理值，如果值是字典
        if isinstance(v, dict):
            new_dict[processed_key] = strip_key_prefix(v)
        else:
            new_dict[processed_key] = v  # 直接复制值（叶子节点）

    return new_dict


# --- 测试代码 ---
if __name__ == "__main__":
    # 示例数据
    test_params = {
        "user_profile": {
            "_id": "user123",  # 带下划线字符串
            "name": "Alice",
            "age": 30,
            "_is_active": True,  # 带下划线布尔值
            "preferences": {
                "theme": "dark",
                "_notifications_on": False,  # 嵌套带下划线布尔
                "language": "en",
                "settings_list": ["a", "b", "c"],  # 不带下划线列表
                "_restricted_set": {1, 2, 3},  # 带下划线集合
                "empty_dict_val": {},  # 空字典
                "_null_value": None,  # 带下划线 None
            },
            "contact_info": {
                "email": "alice@example.com",
                "_phone_number": "123-456-7890",  # 嵌套带下划线字符串
            }
        },
        "system_data": {
            "version":
            "1.0.0",
            "_internal_flag":
            "debug_mode",  # 带下划线字符串
            "config_array":
            jnp.array([10., 20., 30.]),  # JAX 浮点数组不带下划线
            "_metrics_data":
            np.array([[1, 2], [3, 4]]),  # NumPy 整数数组带下划线
            "empty_list": [],  # 空列表
            "_empty_dict": {},  # 空字典带下划线
            "status_tuple": ("ok", 200),  # 元组不带下划线
            "nested_empty": {  # 多层嵌套空字典
                "level1": {
                    "level2": {}
                }
            },
            "_deep_nested_null": {  # 多层嵌套带下划线
                "_data": {
                    "value": None
                }
            },
            "mix_types_list": [  # 列表包含多种类型
                1,
                "text",
                {
                    "_inner_dict_key": "inner_value"
                },  # 列表内的字典
                jnp.array([99]),
                None
            ]
        },
        "_temp_cache": {
            "data": "temp_value"
        },  # 顶层带下划线字典
        "active_session_count": 5,  # 顶层整数
        "conflicting_key":
        "original_value",  # 用于测试 remove_leading_underscores_recursive 的冲突
        "_conflicting_key": "overwritten_value",  # 这个值会覆盖上面的 'conflicting_key'
    }

    print("--- 原始参数结构 ---")
    print(test_params)
    print("-" * 30)

    # --- 测试 create_zero_template 函数 ---
    print("\n--- 测试 create_zero_template ---")
    zero_template_result = reset_to_template(test_params)
    print("生成零模板后的结果:")
    print(zero_template_result)
    print("-" * 30)

    # 测试 create_zero_template 对非字典输入的处理
    print("\n--- create_zero_template 处理非字典输入 ---")
    print(f"传入列表 [1, 2, 3]: {reset_to_template([1, 2, 3])}")
    print(f"传入整数 99: {reset_to_template(99)}")
    print("-" * 30)

    # --- 测试 remove_leading_underscores_recursive 函数 ---
    print("\n--- 测试 remove_leading_underscores_recursive ---")
    # 为了避免修改原始的 test_params，我们先复制一份再进行测试
    params_for_rename = test_params.copy()
    renamed_result = strip_key_prefix(params_for_rename)
    print("移除前导下划线后的结果:")
    print(renamed_result)
    print("-" * 30)

    # 测试 remove_leading_underscores_recursive 对非字典输入的处理
    print("\n--- remove_leading_underscores_recursive 处理非字典输入 ---")
    print(f"传入列表 [1, 2, 3]: {strip_key_prefix([1, 2, 3])}")
    print(f"传入整数 99: {strip_key_prefix(99)}")
    print("-" * 30)

    # 验证 remove_leading_underscores_recursive 是否修改了原始数据 (它会修改副本)
    print(
        "\n--- 验证原始 test_params 未被 remove_leading_underscores_recursive 直接修改 ---"
    )
    print(test_params)
    print("-" * 30)
