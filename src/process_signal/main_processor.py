from src.utils.profiling_utils import static_jit
import jax.numpy as jnp
import jax.lax as lax
from src.indicators.atr import atr_jax
from src.process_signal.core_logic import _calculate_full_state_step
from src.process_signal.signal_handling import _preprocess_signals


@static_jit(static_argnames=("settings"))
def process_trade_signals(_c: dict, signal_result: dict,
                          settings: object) -> dict:
    """
    主函数：处理交易信号，计算仓位和止损止盈历史。

    Args:
        signal_result (dict): 包含原始交易信号的字典。
        _c (dict): 包含OHLCV数据及其他配置的字典。
        settings (object): 包含回测设置（如 unroll 参数）的对象。

    Returns:
        dict: 更新后的 signal_result，包含所有计算结果。
    """
    # 提取K线数据
    high = _c["tohlcv"]["high"]
    low = _c["tohlcv"]["low"]
    close = _c["tohlcv"]["close"]
    open = _c["tohlcv"]["open"]

    # 获取ATR参数和倍数，设置默认值
    atr_period = _c.get("atr_period", jnp.array(14))
    tsl_atr_multiplier = _c.get("tsl_atr_multiplier", jnp.array(2.0))
    sl_atr_multiplier = _c.get("sl_atr_multiplier", jnp.array(2.0))
    tp_atr_multiplier = _c.get("tp_atr_multiplier", jnp.array(2.0))

    # 计算ATR
    raw_atr_values = atr_jax(high,
                             low,
                             close,
                             period=atr_period,
                             unroll=settings.unroll)

    # 提取原始信号
    enter_long_raw = signal_result["enter_long"]
    exit_long_raw = signal_result["exit_long"]
    enter_short_raw = signal_result["enter_short"]
    exit_short_raw = signal_result["exit_short"]

    # 预处理信号
    enter_long, exit_long, enter_short, exit_short = _preprocess_signals(
        enter_long_raw, exit_long_raw, enter_short_raw, exit_short_raw)

    # 准备lax.scan的输入
    inputs = (close, open, raw_atr_values, enter_long, exit_long, enter_short,
              exit_short, tsl_atr_multiplier, sl_atr_multiplier,
              tp_atr_multiplier)

    # 初始化状态
    initial_state = (
        jnp.array(0, dtype=close.dtype),  # position
        jnp.array(0.0, dtype=close.dtype),  # entry_price
        jnp.array(0.0, dtype=close.dtype),  # exit_price
        jnp.array(-jnp.inf, dtype=close.dtype),  # highest_price
        jnp.array(jnp.inf, dtype=close.dtype),  # lowest_price
        jnp.array(0.0, dtype=close.dtype))  # tsl_price

    # 使用lax.scan进行状态更新
    _, (positions, entry_prices, exit_prices, highest_prices, lowest_prices,
        tsl_prices, tsl_triggered_history, sl_triggered_history,
        tp_triggered_history) = lax.scan(_calculate_full_state_step,
                                         initial_state, inputs)

    # 更新signal_result字典
    signal_result.update({
        "position": positions,
        "entry_price": entry_prices,
        "exit_price": exit_prices,
        "highest_price": highest_prices,
        "lowest_price": lowest_prices,
        "tsl_price": tsl_prices,
        "tsl_triggered": tsl_triggered_history,
        "sl_triggered": sl_triggered_history,
        "tp_triggered": tp_triggered_history,
        "atr": raw_atr_values
    })

    return signal_result
