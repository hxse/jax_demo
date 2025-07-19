from run_benchmark import run_benchmark
from src.utils.profiling_utils import print_jax_device_info
from src.utils.jax_cache_config import configure_jax_cache
from src.data_handler.data_loading import load_tohlcv_from_csv, convert_tohlcv_numpy
import typer

micro_path = "database/live/BTC_USDT/15m/BTC_USDT_15m_20230228 160000.csv"
macro_path = "database/live/BTC_USDT/4h/BTC_USDT_4h_20230228 160000.csv"
num = 1
data_size = 40 * 1000
cpu_unroll = 5
gpu_unroll = 9
enable_cpu = True
enable_gpu = True
enable_run_second = False


def main(
    micro_path: str = micro_path,
    macro_path: str = macro_path,
    data_size: int = data_size,
    cpu_unroll: int = cpu_unroll,
    gpu_unroll: int = gpu_unroll,
    enable_cpu: bool = enable_cpu,
    enable_gpu: bool = enable_gpu,
    enable_run_second: bool = enable_run_second,
    num: int = num,
):
    configure_jax_cache(True)

    df_data = load_tohlcv_from_csv(micro_path, data_size=data_size)
    np_data = convert_tohlcv_numpy(df_data)

    df_data2 = load_tohlcv_from_csv(macro_path, data_size=data_size)
    np_data2 = convert_tohlcv_numpy(df_data2)

    result = run_benchmark(np_data,
                           np_data2=np_data2,
                           cpu_unroll=cpu_unroll,
                           gpu_unroll=gpu_unroll,
                           enable_cpu=enable_cpu,
                           enable_gpu=enable_gpu,
                           enable_run_second=enable_run_second,
                           num=num)
    print([k for k, v in result.items() if v != None])
    if enable_cpu:
        print(
            "sma_result length",
            len(result["cpu_result"]["micro"]["indicators"]["sma"]
                ["sma_result"]))
    if enable_gpu:
        print(
            "sma_result length",
            len(result["gpu_result"]["micro"]["indicators"]["sma"]
                ["sma_result"]))


if __name__ == "__main__":
    app = typer.Typer(pretty_exceptions_show_locals=False)
    app.command(help="also ma")(main)
    app.command("ma", hidden=True)(main)
    app()
