import jax
import os
import sys


def configure_jax_cache(enable_cache: bool = True):
    """
    配置 JAX 编译缓存。
    根据操作系统类型设置缓存目录：
    - Linux: ~/jax_cache
    - 其他 (Windows/macOS): ~/.jax_cache
    """
    if not enable_cache:
        print("JAX 编译缓存已禁用。")
        return

    cache_dir = ""
    if sys.platform.startswith("linux"):
        cache_dir = os.path.expanduser("~/jax_cache")
    else:
        cache_dir = os.path.join(os.path.expanduser("~"), ".jax_cache")

    # 确保缓存目录存在
    os.makedirs(cache_dir, exist_ok=True)

    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes",
                      -1)  # 缓存所有大小的条目
    jax.config.update("jax_persistent_cache_min_compile_time_secs",
                      0)  # 缓存所有编译时间大于等于0秒的条目

    print(f"JAX 编译缓存目录已设置为: {cache_dir}")
