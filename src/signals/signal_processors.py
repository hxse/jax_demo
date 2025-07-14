import jax.numpy as jnp
from src.utils.profiling_utils import static_jit
from src.utils.type_checking_utils import check_jnp_array, check_result_dict


@static_jit()
def get_prev_arr(arr: jnp.ndarray) -> jnp.ndarray:
    check_jnp_array(arr, name="Input array", expected_dtype=jnp.bool_)

    # 获取除最后一个元素之外的所有元素，这相当于“向右平移”
    shifted_arr = arr[:-1]
    # 在前面拼接一个False，使数组长度与原数组相同
    return jnp.concatenate([jnp.array([False], dtype=jnp.bool_), shifted_arr])


@static_jit()
def result_reverse(result: dict):
    expected_keys = ["enter_long", "exit_long", "enter_short", "exit_short"]
    check_result_dict(result, expected_keys)

    return {
        "enter_long": result["enter_short"],
        "exit_long": result["exit_short"],
        "enter_short": result["enter_long"],
        "exit_short": result["exit_long"]
    }


@static_jit()
def result_prev(result: dict) -> jnp.ndarray:
    expected_keys = ["enter_long", "exit_long", "enter_short", "exit_short"]
    check_result_dict(result, expected_keys)

    enter_long = result["enter_long"]
    exit_long = result["exit_long"]
    enter_short = result["enter_short"]
    exit_short = result["exit_short"]
    return {
        "enter_long": enter_long & ~get_prev_arr(enter_long),
        "exit_long": exit_long & ~get_prev_arr(exit_long),
        "enter_short": enter_short & ~get_prev_arr(enter_short),
        "exit_short": exit_short & ~get_prev_arr(exit_short),
    }


@static_jit()
def result_enter_prev(result: dict) -> jnp.ndarray:
    expected_keys = ["enter_long", "exit_long", "enter_short", "exit_short"]
    check_result_dict(result, expected_keys)

    enter_long = result["enter_long"]
    exit_long = result["exit_long"]
    enter_short = result["enter_short"]
    exit_short = result["exit_short"]
    return {
        "enter_long": enter_long & ~get_prev_arr(enter_long),
        "exit_long": exit_long,
        "enter_short": enter_short & ~get_prev_arr(enter_short),
        "exit_short": exit_short
    }


@static_jit()
def result_exit_prev(result: dict) -> jnp.ndarray:
    expected_keys = ["enter_long", "exit_long", "enter_short", "exit_short"]
    check_result_dict(result, expected_keys)

    enter_long = result["enter_long"]
    exit_long = result["exit_long"]
    enter_short = result["enter_short"]
    exit_short = result["exit_short"]
    return {
        "enter_long": enter_long,
        "exit_long": exit_long & ~get_prev_arr(exit_long),
        "enter_short": enter_short,
        "exit_short": exit_short & ~get_prev_arr(exit_short)
    }


@static_jit()
def no_result_neutral(result: dict) -> dict:
    '''
    enter用&, exit用|, 所以enter用True, exit用False, 确保不干扰后续的连接
    '''
    expected_keys = ["enter_long", "exit_long", "enter_short", "exit_short"]
    check_result_dict(result, expected_keys)

    output_shape = result["enter_long"].shape
    # output_dtype = result["enter_long"].dtype
    output_dtype = jnp.bool_
    return {
        "enter_long": jnp.full(output_shape, True, dtype=output_dtype),
        "exit_long": jnp.full(output_shape, False, dtype=output_dtype),
        "enter_short": jnp.full(output_shape, True, dtype=output_dtype),
        "exit_short": jnp.full(output_shape, False, dtype=output_dtype),
    }


@static_jit()
def no_result_all_False(result: dict) -> dict:
    '''
    enter用&, exit用|, 所以enter用True, exit用False, 确保不干扰后续的连接
    '''
    expected_keys = ["enter_long", "exit_long", "enter_short", "exit_short"]
    check_result_dict(result, expected_keys)

    output_shape = result["enter_long"].shape
    # output_dtype = result["enter_long"].dtype
    output_dtype = jnp.bool_
    return {
        "enter_long": jnp.full(output_shape, False, dtype=output_dtype),
        "exit_long": jnp.full(output_shape, False, dtype=output_dtype),
        "enter_short": jnp.full(output_shape, False, dtype=output_dtype),
        "exit_short": jnp.full(output_shape, False, dtype=output_dtype),
    }


@static_jit()
def origin_result(result: dict) -> dict:
    expected_keys = ["enter_long", "exit_long", "enter_short", "exit_short"]
    check_result_dict(result, expected_keys)
    return result
