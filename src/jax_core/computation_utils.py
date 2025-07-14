from src.utils.jax_utils import block_jax_output
from src.utils.profiling_utils import timer


@timer(name="run_computation")  # Apply timer directly here with a custom name
def run_computation(func, data_on_device, settings):
    """在设备上运行 JAX 计算并等待完成。"""
    # 这里 data_on_device 已经是 JAX 数组并已在目标设备上
    result = func(data_on_device, settings)
    block_jax_output(result)
    return result
