from .signal_template import sma_simple, sma_close, rsi_simple, psar_simple
from src.utils.profiling_utils import static_jit


@static_jit()
def sma_simple_wrapper(_d, _c, _i):
    return sma_simple(_i["sma"]["sma_result"], _i["sma2"]["sma2_result"],
                      _d["close"])


@static_jit()
def sma_close_wrapper(_d, _c, _i):
    return sma_close(_i["sma"]["sma_result"], _i["sma2"]["sma2_result"],
                     _d["close"])


@static_jit()
def rsi_simple_wrapper(_d, _c, _i):
    return rsi_simple(_d["close"], _c["sma"]["period"])


@static_jit()
def psar_simple_wrapper(_d, _c, _i):
    return psar_simple(_i["psar"]["psar_long_result"],
                       _i["psar"]["psar_short_result"])


indicator_template = {
    "sma": [sma_simple_wrapper, sma_close_wrapper],
    "rsi": [rsi_simple_wrapper],
    "psar": [psar_simple_wrapper]
}

indicator_names = ["sma", "sma2", "rsi", "atr", "psar"]  # 技术指标列表

indicator_names_lite = [
    i for i in indicator_names if i not in ["sma2", "atr"]
]  # 信号生成列表
