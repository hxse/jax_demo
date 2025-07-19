from run_benchmark import run_benchmark
from src.utils.profiling_utils import print_jax_device_info
from src.utils.jax_cache_config import configure_jax_cache
from src.data_handler.data_loading import load_tohlcv_from_csv, convert_tohlcv_numpy

if __name__ == "__main__":
    configure_jax_cache(True)

    micro_path = "database/live/BTC_USDT/15m/BTC_USDT_15m_20230228 160000.csv"

    data_size = 35040

    df_data = load_tohlcv_from_csv(micro_path, data_size=data_size)
    np_data = convert_tohlcv_numpy(df_data)

    cpu_second_run_times = []
    cpu_unroll_params = []
    gpu_second_run_times = []
    gpu_unroll_params = []

    # find_unroll = True
    for i in range(1, 11):
        benchmark_results = run_benchmark(np_data,
                                          cpu_unroll=i,
                                          gpu_unroll=i,
                                          enable_cpu=True,
                                          enable_gpu=True,
                                          enable_run_second=True)
        cpu_time = benchmark_results['second_run_cpu_time']
        gpu_time = benchmark_results['second_run_gpu_time']
        if cpu_time is not None:
            cpu_second_run_times.append(cpu_time)
            cpu_unroll_params.append(i)
        if gpu_time is not None:
            gpu_second_run_times.append(gpu_time)
            gpu_unroll_params.append(i)
        benchmark_results = None

    if cpu_second_run_times:
        best_cpu_unroll, min_cpu_time = min(zip(cpu_unroll_params,
                                                cpu_second_run_times),
                                            key=lambda x: x[1])
        print(
            f"\nCPU 最佳 unroll 参数: {best_cpu_unroll}, 最小耗时: {min_cpu_time:.4f} 秒"
        )
    else:
        print("\n没有记录到 CPU 第二次运行时间。")

    if gpu_second_run_times:
        best_gpu_unroll, min_gpu_time = min(zip(gpu_unroll_params,
                                                gpu_second_run_times),
                                            key=lambda x: x[1])
        print(
            f"\nGPU 最佳 unroll 参数: {best_gpu_unroll}, 最小耗时: {min_gpu_time:.4f} 秒"
        )
    else:
        print("\n没有记录到 GPU 第二次运行时间。")
