import time

_cold_start_begin_time = time.time()

import sys
from pathlib import Path

root_path = next((p for p in Path(__file__).resolve().parents
                  if (p / "pyproject.toml").is_file()), None)
if root_path:
    sys.path.insert(0, str(root_path))

import os

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

import jax

jax.config.update("jax_enable_x64", True)

from jax import vmap

from src.utils.config_utils import get_config_vmap, settings
from src.utils.profiling_utils import print_jax_device_info

from src.jax_core.computation_utils import run_computation
from src.jax_core.strategy_function import run_strategy
import numpy as np


def run_benchmark(np_data,
                  np_data2=None,
                  indicator_params=None,
                  indicator_in_axes=None,
                  cpu_unroll=2,
                  gpu_unroll=8,
                  enable_cpu=True,
                  enable_gpu=False,
                  enable_run_second=False):

    # Initialize all result and time variables to None
    run_cpu_time = None
    run_gpu_time = None
    cpu_result = None
    gpu_result = None
    second_run_cpu_time = None
    second_run_gpu_time = None
    second_cpu_result = None
    second_gpu_result = None

    print("cpu_unroll", cpu_unroll, "gpu_unroll", gpu_unroll)

    cpu_settings = settings(unroll=cpu_unroll)
    gpu_settings = settings(unroll=gpu_unroll)

    if np_data2 is None:
        np_data2 = np.full_like(np_data[:10], np.nan)

    _ = get_config_vmap(np_data.copy(), np_data2.copy(), indicator_params,
                        indicator_in_axes)
    data_vmap_cpu, data_in_axes_cpu = _
    _ = get_config_vmap(np_data.copy(), np_data2.copy(), indicator_params,
                        indicator_in_axes)
    data_vmap_gpu, data_in_axes_gpu = _

    vmap_function_cpu = vmap(
        run_strategy,
        in_axes=(data_in_axes_cpu, None),
    )
    vmap_function_gpu = vmap(run_strategy, in_axes=(data_in_axes_gpu, None))

    for run_idx in range(2 if enable_run_second else 1):
        run_label = "第一次运行" if run_idx == 0 else "第二次运行"
        print(f"\n--- {run_label} ---")

        # CPU Run
        if not bool(cpu_devices):
            print("CPU 不可用，跳过")
        elif enable_cpu:
            print(f"\n--- vmap 在 CPU 上的执行 ({run_label}) ---")
            data_on_cpu_device = jax.device_put(data_vmap_cpu, cpu_devices[0])
            if run_idx == 0:
                cpu_result, run_cpu_time = run_computation(
                    vmap_function_cpu, data_on_cpu_device, cpu_settings)
            elif run_idx == 1:
                second_cpu_result, second_run_cpu_time = run_computation(
                    vmap_function_cpu, data_on_cpu_device, cpu_settings)

        # GPU Run
        if not bool(gpu_devices):
            print("CUDA 不可用，跳过")
        elif enable_gpu:
            print(f"\n--- vmap 在 GPU 上的执行 ({run_label}) ---")
            data_on_gpu_device = jax.device_put(data_vmap_gpu, gpu_devices[0])
            if run_idx == 0:
                gpu_result, run_gpu_time = run_computation(
                    vmap_function_gpu, data_on_gpu_device, gpu_settings)
            elif run_idx == 1:
                second_gpu_result, second_run_gpu_time = run_computation(
                    vmap_function_gpu, data_on_gpu_device, gpu_settings)

    return {
        "run_cpu_time": run_cpu_time,
        "run_gpu_time": run_gpu_time,
        "cpu_result": cpu_result,
        "gpu_result": gpu_result,
        "second_run_cpu_time": second_run_cpu_time,
        "second_run_gpu_time": second_run_gpu_time,
        "second_cpu_result": second_cpu_result,
        "second_gpu_result": second_gpu_result,
    }


print(f"冷启动时间: {time.time() - _cold_start_begin_time:.4f} 秒")

cpu_devices, gpu_devices = print_jax_device_info()
