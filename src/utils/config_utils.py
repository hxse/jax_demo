import jax.numpy as jnp

from dataclasses import dataclass, field
from jax.tree_util import register_dataclass
from src.utils.jax_utils import reset_to_template, strip_key_prefix


def indicator_params_template(period=[14, 15, 1],
                              diff=[50],
                              af0=[0.02, 0.03, 0.01],
                              af_step=[0.02, 0.03, 0.01],
                              max_af=[0.2, 0.3, 0.1],
                              enable_list=["sma"],
                              enable_enter_prev=["sma"],
                              enable_exit_prev=["sma"],
                              enable_reverse=[]):
    '''
    不需要vmap矢量化分配的参数, 用下划线_开头
    下划线会在后续外部自动移除, 所以调用的时候不要带下划线
    '''
    if "sma" in enable_list:
        enable_list.append("sma2")

    indicator_params = {
        "sma": {
            "period": jnp.arange(period[0], period[1], period[2]),
            "_template_idx": jnp.array(0),
            "_enable": jnp.array(False),
            "_enable_enter_prev": jnp.array(False),
            "_enable_exit_prev": jnp.array(False),
            "_enable_reverse": jnp.array(False)
        },
        "sma2": {
            "period":
            jnp.arange(period[0] + diff[0], period[1] + diff[0],
                       period[2] + diff[0]),
            "_enable":
            jnp.array(False),
        },
        "rsi": {
            "period": jnp.arange(period[0], period[1], period[2]),
            "_threshold": jnp.array(30),
            "_template_idx": jnp.array(0),
            "_enable": jnp.array(False),
            "_enable_enter_prev": jnp.array(False),
            "_enable_exit_prev": jnp.array(False),
            "_enable_reverse": jnp.array(False),
        },
        "atr": {
            "period": jnp.arange(period[0], period[1], period[2]),
            "_template_idx": jnp.array(0),
            "_enable": jnp.array(False),
            "_enable_enter_prev": jnp.array(False),
            "_enable_exit_prev": jnp.array(False),
            "_enable_reverse": jnp.array(False),
        },
        "psar": {
            "af0": jnp.arange(af0[0], af0[1], af0[2]),
            "af_step": jnp.arange(af_step[0], af_step[1], af_step[2]),
            "max_af": jnp.arange(max_af[0], max_af[1], max_af[2]),
            "_template_idx": jnp.array(0),
            "_enable": jnp.array(False),
            "_enable_enter_prev": jnp.array(False),
            "_enable_exit_prev": jnp.array(False),
            "_enable_reverse": jnp.array(False),
        },
    }

    for k, v in indicator_params.items():
        if k in enable_list:
            v["_enable"] = jnp.array(True)
        if k in enable_enter_prev:
            v["_enable_enter_prev"] = jnp.array(True)
        if k in enable_exit_prev:
            v["_enable_exit_prev"] = jnp.array(True)
        if k in enable_reverse:
            v["_enable_reverse"] = jnp.array(True)

    return indicator_params


# --- 1. 定义统一的配置 dataclass ---
@register_dataclass
@dataclass(frozen=True)
class settings:
    unroll: int = field(metadata=dict(static=True))


def get_config_vmap(np_data,
                    np_data2,
                    indicator_params=None,
                    indicator_in_axes=None,
                    indicator_params2=None,
                    indicator_in_axes2=None):
    """
    将中间配置列表转换为适用于 JAX vmap 的数据字典。

    Args:
        intermediate_config_list (list): 包含配置字典的列表，格式如下：
                                         [{"rsi": {"period": i}, {"sma": {"period": i}}, ...]
        close: 用于 'close' 键的值，可以是 JAX 数组或其他数据。

    Returns:
        dict: 转换后的数据字典，格式如下：
              {
                  "close": close,
                  "rsi": {"period": jnp.array([...])},
                  "sma": {"period": jnp.array([...])},
              }
    """

    if indicator_params == None:
        indicator_params = indicator_params_template()
    if indicator_params2 == None:
        indicator_params2 = indicator_params_template()

    if indicator_in_axes == None:
        indicator_in_axes = reset_to_template(indicator_params)

    if indicator_in_axes2 == None:
        indicator_in_axes2 = reset_to_template(indicator_params2)

    indicator_params = strip_key_prefix(indicator_params)
    indicator_params2 = strip_key_prefix(indicator_params2)
    indicator_in_axes = strip_key_prefix(indicator_in_axes)
    indicator_in_axes2 = strip_key_prefix(indicator_in_axes2)

    data_vmap = {
        "micro": {
            "tohlcv": {
                "time": jnp.array(np_data[:, 0]),
                "open": jnp.array(np_data[:, 1]),
                "high": jnp.array(np_data[:, 2]),
                "low": jnp.array(np_data[:, 3]),
                "close": jnp.array(np_data[:, 4]),
                "volume": jnp.array(np_data[:, 5]),
            },
            **indicator_params
        },
        "macro": {
            "tohlcv": {
                "time": jnp.array(np_data2[:, 0]),
                "open": jnp.array(np_data2[:, 1]),
                "high": jnp.array(np_data2[:, 2]),
                "low": jnp.array(np_data2[:, 3]),
                "close": jnp.array(np_data2[:, 4]),
                "volume": jnp.array(np_data2[:, 5]),
            },
            **indicator_params2
        }
    }

    data_in_axes = {
        "micro": {
            "tohlcv": {
                "time": None,
                "open": None,
                "high": None,
                "low": None,
                "close": None,
                "volume": None,
            },
            **indicator_in_axes
        },
        "macro": {
            "tohlcv": {
                "time": None,
                "open": None,
                "high": None,
                "low": None,
                "close": None,
                "volume": None,
            },
            **indicator_in_axes2
        }
    }

    return data_vmap, data_in_axes
