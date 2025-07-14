from src.utils.profiling_utils import static_jit
from src.jax_core.indicator_functions import calculate_indicators
from src.jax_core.signal_generator import generate_signal
from src.process_signal.main_processor import process_trade_signals


@static_jit(static_argnames=("settings"))
def run_strategy(config: dict, settings: object) -> dict:

    final_output = {"micro": {}, "macro": {}}
    for data_mode in ["micro", "macro"]:
        _c = config[data_mode]
        indicators_result = calculate_indicators(_c, settings)
        signal_result = generate_signal(_c, indicators_result, settings)
        processed_signal_result = process_trade_signals(
            _c, signal_result, settings)

        final_output[data_mode]["indicators"] = indicators_result
        final_output[data_mode]["signal"] = signal_result
        final_output[data_mode]["processed_signal"] = processed_signal_result

    return final_output
