import jax.numpy as jnp
from dataclasses import dataclass, replace
from jax.tree_util import register_dataclass


@register_dataclass
@dataclass(frozen=False)
class TradeState:
    position: jnp.ndarray
    entry_price: jnp.ndarray
    exit_price: jnp.ndarray
    highest_price: jnp.ndarray
    lowest_price: jnp.ndarray
    atr_sl_price: jnp.ndarray
    atr_tp_price: jnp.ndarray
    atr_tsl_price: jnp.ndarray
    holding_bars: jnp.ndarray


@register_dataclass
@dataclass(frozen=False)
class TradeStateLast:
    position: jnp.ndarray


@register_dataclass
@dataclass(frozen=False)
class TradeIntermediate:
    trade_price: jnp.ndarray
    atr_tsl_triggered: jnp.ndarray
    atr_sl_triggered: jnp.ndarray
    atr_tp_triggered: jnp.ndarray
    bar_index: jnp.ndarray
    min_holding_period_triggered: jnp.ndarray
    max_holding_period_triggered: jnp.ndarray
    should_exit_long: jnp.ndarray
    should_exit_short: jnp.ndarray


@register_dataclass
@dataclass(frozen=False)
class TradeInputs:
    time: jnp.ndarray
    open: jnp.ndarray
    high: jnp.ndarray
    low: jnp.ndarray
    close: jnp.ndarray
    volume: jnp.ndarray
    raw_atr_values: jnp.ndarray
    enter_long: jnp.ndarray
    exit_long: jnp.ndarray
    enter_short: jnp.ndarray
    exit_short: jnp.ndarray


@register_dataclass
@dataclass(frozen=False)
class TradeOutputs:
    positions: jnp.ndarray
    entry_prices: jnp.ndarray
    exit_prices: jnp.ndarray
    highest_prices: jnp.ndarray
    lowest_prices: jnp.ndarray
    atr_tsl_prices: jnp.ndarray
    atr_tsl_triggered_history: jnp.ndarray
    atr_sl_triggered_history: jnp.ndarray
    atr_tp_triggered_history: jnp.ndarray


@register_dataclass
@dataclass(frozen=False)
class TradeParams:
    # 这些设置参数在扫描过程中不会改变
    atr_tsl_multiplier: jnp.ndarray
    atr_sl_multiplier: jnp.ndarray
    atr_tp_multiplier: jnp.ndarray
    min_holding_period: jnp.ndarray
    max_holding_period: jnp.ndarray
