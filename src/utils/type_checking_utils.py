import jax.numpy as jnp


def check_jnp_array(arr: jnp.ndarray,
                    name: str = "Input",
                    expected_dtype: jnp.dtype = None):
    """
    检查 arr 是否为 jnp.ndarray 类型，并可选地检查其 dtype。
    如果检查失败，抛出 TypeError。
    """
    if not isinstance(arr, jnp.ndarray):
        raise TypeError(f"{name} 必须是 jnp.ndarray 类型，但得到的是 {type(arr)}。")
    if expected_dtype is not None and arr.dtype != expected_dtype:
        raise TypeError(
            f"{name} 的 dtype 必须是 {expected_dtype}，但得到的是 {arr.dtype}。")


def check_result_dict(result: dict,
                      expected_keys: list,
                      expected_dtype: jnp.dtype = None):
    """
    检查 result 是否为字典类型，包含所有预期键，并且每个键对应的值都是 jnp.ndarray 类型。
    如果提供了 expected_dtype，则检查每个键对应的值的 dtype 是否与 expected_dtype 匹配。
    如果检查失败，抛出 TypeError 或 ValueError。
    """
    if not isinstance(result, dict):
        raise TypeError(f"结果必须是字典类型，但得到的是 {type(result)}。")

    for key in expected_keys:
        if key not in result:
            raise ValueError(f"结果字典缺少预期键: '{key}'。")

        value = result[key]
        if not isinstance(value, jnp.ndarray):
            raise TypeError(
                f"键 '{key}' 对应的值必须是 jnp.ndarray 类型，但得到的是 {type(value)}。")

        if expected_dtype is not None and value.dtype != expected_dtype:
            raise TypeError(
                f"键 '{key}' 对应的值的 dtype 必须是 {expected_dtype}，但得到的是 {value.dtype}。"
            )
