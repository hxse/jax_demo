from src.indicators.sma import sma_jax, no_sma_jax
from src.indicators.rsi import rsi_jax, no_rsi_jax
from src.indicators.tr import tr_jax, no_tr_jax
from src.indicators.atr import atr_jax, no_atr_jax
from src.indicators.psar import psar_jax, no_psar_jax
from src.utils.profiling_utils import static_jit


@static_jit()
def sma_func(_d, _c):
    return {"sma_result": sma_jax(_d["close"], _c["sma"]["period"])}


@static_jit()
def no_sma_func(_d, _c):
    return {"sma_result": no_sma_jax(_d["close"], _c["sma"]["period"])}


@static_jit()
def sma2_func(_d, _c):
    return {"sma2_result": sma_jax(_d["close"], _c["sma2"]["period"])}


@static_jit()
def no_sma2_func(_d, _c):
    return {"sma2_result": no_sma_jax(_d["close"], _c["sma2"]["period"])}


@static_jit(static_argnames=("unroll"))
def rsi_func(_d, _c, unroll):
    return {"rsi_result": rsi_jax(_d["close"], _c["rsi"]["period"], unroll)}


@static_jit(static_argnames=("unroll"))
def no_rsi_func(_d, _c, unroll):
    return {"rsi_result": no_rsi_jax(_d["close"], _c["rsi"]["period"], unroll)}


@static_jit()
def tr_func(_d, _c):
    return {"tr_result": tr_jax(_d["high"], _d["low"], _d["close"])}


@static_jit()
def no_tr_func(_d, _c):
    return {"tr_result": no_tr_jax(_d["high"], _d["low"], _d["close"])}


@static_jit(static_argnames=("unroll"))
def atr_func(_d, _c, unroll):
    return {
        "atr_result":
        atr_jax(_d["high"], _d["low"], _d["close"], _c["atr"]["period"],
                unroll)
    }


@static_jit(static_argnames=("unroll"))
def no_atr_func(_d, _c, unroll):
    return {
        "atr_result":
        no_atr_jax(_d["high"], _d["low"], _d["close"], _c["atr"]["period"],
                   unroll)
    }


@static_jit(static_argnames=("unroll"))
def psar_func(_d, _c, unroll):
    (psar_long_result, psar_short_result, psar_af_result,
     psar_reversal_result) = psar_jax(_d["high"], _d["low"], _d["close"],
                                      _c["psar"]["af0"], _c["psar"]["af_step"],
                                      _c["psar"]["max_af"], unroll)
    return {
        "psar_long_result": psar_long_result,
        "psar_short_result": psar_short_result,
        "psar_af_result": psar_af_result,
        "psar_reversal_result": psar_reversal_result,
    }


@static_jit(static_argnames=("unroll"))
def no_psar_func(_d, _c, unroll):
    (psar_long_result, psar_short_result, psar_af_result,
     psar_reversal_result) = no_psar_jax(_d["high"], _d["low"], _d["close"],
                                         _c["psar"]["af0"],
                                         _c["psar"]["af_step"],
                                         _c["psar"]["max_af"], unroll)
    return {
        "psar_long_result": psar_long_result,
        "psar_short_result": psar_short_result,
        "psar_af_result": psar_af_result,
        "psar_reversal_result": psar_reversal_result,
    }


indicator_functions = {
    "sma": {
        "func": sma_func,
        "no_func": no_sma_func,
    },
    "sma2": {
        "func": sma2_func,
        "no_func": no_sma2_func,
    },
    "rsi": {
        "func": rsi_func,
        "no_func": no_rsi_func,
        "unroll": True
    },
    "atr": {
        "func": atr_func,
        "no_func": no_atr_func,
        "unroll": True
    },
    "psar": {
        "func": psar_func,
        "no_func": no_psar_func,
        "unroll": True
    }
}
