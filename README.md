# use wsl
  * 直接把项目`git clone`到wsl内部, 然后`code .`打开项目
  * `uv run python src/examples/run_simple_benchmark.py ma --num 1 --enable-gpu --enable-run-second --cpu-unroll 4 --gpu-unroll 8`
# todo
  * 目前,技术指标计算,指标探索模块,和指标信号生成都弄好了
  * `src/process_signal`指标信号进一步处理, 添加atr_sl tp tsl还没写完, 先不写了, 去试试手动交易, 以后有空再写吧
# jax.lax.scan的性能问题
  * <https://github.com/jax-ml/jax/issues/2491#issuecomment-653775607>
  * scan在cpu和tpu上都性能正常,在gpu上性能会有问题,只能盼着官方优化了
