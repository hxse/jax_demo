import jax.numpy as jnp
from src.utils.profiling_utils import static_jit


@static_jit()
def sma_simple(sma: jnp.ndarray, sma2: jnp.ndarray,
               close: jnp.ndarray) -> dict:
    return {
        "enter_long": (sma > sma2),
        "exit_long": (sma < sma2),
        "enter_short": (sma < sma2),
        "exit_short": (sma > sma2)
    }


@static_jit()
def sma_close(sma: jnp.ndarray, sma2: jnp.ndarray, close: jnp.ndarray) -> dict:
    return {
        "enter_long": (sma > sma2) & (close > sma),
        "exit_long": (sma < sma2) | (close < sma),
        "enter_short": (sma < sma2) & (close < sma),
        "exit_short": (sma > sma2) | (close > sma)
    }


@static_jit()
def rsi_simple(rsi, threshold):
    return {
        "enter_long": rsi > threshold,
        "exit_long": rsi < 100 - threshold,
        "enter_short": rsi < 100 - threshold,
        "exit_short": rsi > threshold
    }


@static_jit()
def psar_simple(psar_long: jnp.ndarray, psar_short: jnp.ndarray):
    return {
        "enter_long": ~jnp.isnan(psar_long),
        "exit_long": ~jnp.isnan(psar_short),
        "enter_short": ~jnp.isnan(psar_short),
        "exit_short": ~jnp.isnan(psar_long)
    }
