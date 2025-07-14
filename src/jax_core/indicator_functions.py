import jax
import jax.numpy as jnp
from src.utils.profiling_utils import static_jit
from src.indicators.sma import sma_jax, no_sma_jax
from src.indicators.rsi import rsi_jax, no_rsi_jax
from src.signals.signal_wrapper import indicator_names, indicator_names_lite
from src.indicators.indicators_wrapper import indicator_functions
from src.utils.jax_utils import get_adapted_indicator_functions


@static_jit(static_argnames=("settings"))
def calculate_indicators(_c: dict, settings: object) -> dict:
    """
    生成技术指标
    """
    current_mode_results = {}
    for name in indicator_names:

        _r = indicator_functions[name]
        func, no_func = get_adapted_indicator_functions(_r, settings)

        indicator_output_array = jax.lax.cond(_c[name]["enable"], func,
                                              no_func, (_c["tohlcv"], _c))

        current_mode_results[name] = indicator_output_array
    result = {name: current_mode_results[name] for name in indicator_names}

    return result
