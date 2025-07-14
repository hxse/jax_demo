import jax
import jax.numpy as jnp
from src.utils.profiling_utils import static_jit
from src.signals.signal_processors import (result_reverse, result_prev,
                                           result_enter_prev, result_exit_prev,
                                           no_result_neutral,
                                           no_result_all_False, origin_result)

from src.signals.signal_wrapper import indicator_template, indicator_names, indicator_names_lite


@static_jit(static_argnames=("settings"))
def generate_signal(_c: dict, _i: dict, settings: object) -> dict:
    """
    生成交易信号
    """
    current_mode_results = {}
    for name in indicator_names_lite:
        enable = _c[name]["enable"]
        template_idx = _c[name]["template_idx"]
        reverse = _c[name]["enable_reverse"]
        enter_prev = _c[name]["enable_enter_prev"]
        exit_prev = _c[name]["enable_exit_prev"]

        res = jax.lax.switch(template_idx, indicator_template[name],
                             *(_c["tohlcv"], _c, _i))
        res = jax.lax.cond(reverse, result_reverse, origin_result, res)
        res = jax.lax.cond(enter_prev, result_enter_prev, origin_result, res)
        res = jax.lax.cond(exit_prev, result_exit_prev, origin_result, res)
        res = jax.lax.cond(enable, origin_result, no_result_neutral, res)
        current_mode_results[name] = res

    # enter进场条件用&连接, exit离场条件用|连接
    res = {}
    for i in ["enter_long", "exit_long", "enter_short", "exit_short"]:
        all_enter_long_signals = jnp.array(
            [current_mode_results[name][i] for name in indicator_names_lite],
            dtype=jnp.bool_)
        if i in ["enter_long", "enter_short"]:
            res[i] = jnp.all(all_enter_long_signals, axis=0)
        else:
            res[i] = jnp.any(all_enter_long_signals, axis=0)

    # 检查当前 data_mode 下是否所有入场指标都被禁用
    enable_states_for_all_indicators = jnp.array(
        [_c[name]["enable"] for name in indicator_names_lite], dtype=jnp.bool_)
    all_enable_disabled = jnp.all(~enable_states_for_all_indicators)
    # 如果所有入场指标都禁用，则 enter_long 和 enter_short 设为全 False
    res = jax.lax.cond(all_enable_disabled, no_result_all_False, origin_result,
                       res)

    return res
