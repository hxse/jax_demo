from run_benchmark import run_benchmark
from src.utils.profiling_utils import print_jax_device_info
from src.utils.jax_cache_config import configure_jax_cache
from src.data_handler.data_loading import load_tohlcv_from_csv, convert_tohlcv_numpy

if __name__ == "__main__":
    configure_jax_cache(True)
    cpu_devices, gpu_devices = print_jax_device_info()

    micro_path = "database/live/BTC_USDT/15m/BTC_USDT_15m_20230228 160000.csv"
    macro_path = "database/live/BTC_USDT/4h/BTC_USDT_4h_20230228 160000.csv"

    df_data = load_tohlcv_from_csv(micro_path, data_size=None)
    np_data = convert_tohlcv_numpy(df_data)

    df_data2 = load_tohlcv_from_csv(macro_path, data_size=None)
    np_data2 = convert_tohlcv_numpy(df_data2)

    result = run_benchmark(np_data,
                           np_data2=np_data2,
                           cpu_unroll=2,
                           gpu_unroll=8,
                           enable_cpu=True,
                           enable_gpu=True,
                           enable_run_second=False)
    print([k for k, v in result.items() if v != None])
    import pdb
    pdb.set_trace()
