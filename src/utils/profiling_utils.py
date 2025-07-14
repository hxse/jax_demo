import jax
import time
from functools import wraps
from jax import jit


def static_jit(*args, **kwargs):
    """
    一个 JIT 装饰器，用于将函数的指定参数标记为静态。
    例如：@static_jit(static_argnums=(2,))
    """

    def decorator(func):
        # 将命名参数传递给 jit
        jitted_func = jit(func, *args, **kwargs)

        @wraps(func)
        def wrapper(*args, **kwargs):
            return jitted_func(*args, **kwargs)

        return wrapper

    return decorator


# 修改 timer 装饰器，使其可以接收一个可选的 name 参数
def timer(name=None):
    """
    一个灵活的计时器，既可以作为装饰器也可以作为包装器。
    当作为装饰器使用时，它不接收参数。
    当直接调用以获取一个计时上下文时，它接收一个可选的 name 参数。
    """
    if callable(name):  # 如果 timer 直接修饰函数，name 会是那个函数
        func = name

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            # 使用被装饰函数的名称
            print(f"函数 '{func.__name__}' 执行时间: {elapsed_time:.6f} 秒")
            return result, elapsed_time

        return wrapper
    else:  # 如果 timer(name="...") 这种方式调用

        def wrapper_func(func):

            @wraps(func)
            def inner_wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                elapsed_time = end_time - start_time
                # 使用传入的 name 参数，如果 name 为 None 则使用函数名
                display_name = name if name is not None else func.__name__
                print(f"'{display_name}' 执行时间: {elapsed_time:.6f} 秒")
                return result, elapsed_time

            return inner_wrapper

        return wrapper_func


def print_jax_device_info():
    print("--- JAX 设备检测 ---")
    print(f"JAX64 : {jax.config.jax_enable_x64}")

    print("\n--- 详细设备信息 ---")
    cpu_devices = jax.devices("cpu")
    print(f"CPU 设备: {cpu_devices}")
    gpu_devices = jax.devices("gpu")
    print(f"GPU 设备: {gpu_devices}")

    print(f"\nJAX 默认设备: {jax.default_backend()}")
    print("--------------------")

    print(f"CUDA 可用: {bool(gpu_devices)}")
    print("--------------------")

    return cpu_devices, gpu_devices
