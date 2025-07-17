from src.utils.profiling_utils import static_jit
import jax.numpy as jnp
import jax.lax as lax
from src.indicators.atr import atr_jax
from src.backtester.core_logic import _calculate_full_state_step, _update_bar_index
from src.backtester.signal_handling import _preprocess_signals
from src.backtester.trade_state import TradeState, TradeInputs, TradeOutputs, TradeIntermediate, TradeParams, TradeStateLast


@static_jit(static_argnames=("settings"))
def process_trade_signals(_c: dict, signal_result: dict,
                          settings: object) -> dict:
    """
    主函数：处理交易信号，计算仓位和止损止盈历史。

    Args:
        signal_result (dict): 包含原始交易信号的字典。
        _c (dict): 包含OHLCV数据（tohlcv 键）和ATR相关配置（atr_period, tsl_atr_multiplier 等）的字典。
        settings (object): 包含回测设置的对象，例如 settings.unroll 用于控制 lax.scan 的展开行为。

    Returns:
        dict: 更新后的 signal_result，包含所有计算结果。
    """
    # 提取K线数据
    time = _c["tohlcv"]["time"]
    open = _c["tohlcv"]["open"]
    high = _c["tohlcv"]["high"]
    low = _c["tohlcv"]["low"]
    close = _c["tohlcv"]["close"]
    volume = _c["tohlcv"]["volume"]

    # 获取ATR参数和倍数，设置默认值
    atr_period = _c.get("atr_period", jnp.array(14))
    atr_tsl_multiplier = _c.get("atr_tsl_multiplier", jnp.array(2.0))
    atr_sl_multiplier = _c.get("atr_sl_multiplier", jnp.array(2.0))
    atr_tp_multiplier = _c.get("atr_tp_multiplier", jnp.array(2.0))
    min_holding_period = _c.get("min_holding_period", jnp.array(-jnp.inf))
    max_holding_period = _c.get("max_holding_period", jnp.array(jnp.inf))

    trade_params = TradeParams(atr_tsl_multiplier=atr_tsl_multiplier,
                               atr_sl_multiplier=atr_sl_multiplier,
                               atr_tp_multiplier=atr_tp_multiplier,
                               min_holding_period=min_holding_period,
                               max_holding_period=max_holding_period)

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
    inputs = TradeInputs(time=time,
                         open=open,
                         high=high,
                         low=low,
                         close=close,
                         volume=volume,
                         raw_atr_values=raw_atr_values,
                         enter_long=enter_long,
                         exit_long=exit_long,
                         enter_short=enter_short,
                         exit_short=exit_short)

    # 确保所有 JAX 数组的数据类型与 close 价格一致，以保持计算精度。
    initial_state = TradeState(
        position=jnp.array(0),
        entry_price=jnp.array(0.0),
        exit_price=jnp.array(0.0),
        highest_price=jnp.array(jnp.nan),
        lowest_price=jnp.array(jnp.nan),
        atr_sl_price=jnp.array(0.0),
        atr_tp_price=jnp.array(0.0),
        atr_tsl_price=jnp.array(0.0),
        holding_bars=jnp.array(0),
    )
    initial_state_last = TradeStateLast(position=jnp.array(0), )

    initial_intermediate = TradeIntermediate(
        trade_price=jnp.array(0.0),
        atr_tsl_triggered=jnp.array(False),
        atr_sl_triggered=jnp.array(False),
        atr_tp_triggered=jnp.array(False),
        bar_index=jnp.array(-1),
        min_holding_period_triggered=jnp.array(True),
        max_holding_period_triggered=jnp.array(False),
        should_exit_long=jnp.array(False),
        should_exit_short=jnp.array(False))

    # 使用 lax.scan 进行高效迭代处理时序数据，避免 Python 循环以实现 JIT 优化。
    (final_state, final_stateLast,
     final_intermediate), trade_outputs = lax.scan(
         lambda carry, x: _calculate_full_state_step(carry, x, trade_params),
         (initial_state, initial_state_last, initial_intermediate),
         inputs,
         unroll=settings.unroll)

    # 更新signal_result字典
    signal_result.update({
        "position": trade_outputs.positions,
        "entry_price": trade_outputs.entry_prices,
        "exit_price": trade_outputs.exit_prices,
        "highest_price": trade_outputs.highest_prices,
        "lowest_price": trade_outputs.lowest_prices,
        "atr_tsl_price": trade_outputs.atr_tsl_prices,
        "atr_tsl_triggered": trade_outputs.atr_tsl_triggered_history,
        "atr_sl_triggered": trade_outputs.atr_sl_triggered_history,
        "atr_tp_triggered": trade_outputs.atr_tp_triggered_history,
        "atr": raw_atr_values
    })

    return signal_result
