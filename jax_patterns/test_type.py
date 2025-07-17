import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

a = jnp.array(0.0)
print(f"a 的类型: {a.dtype}")  # 0.0 是浮点数

b = jnp.array(0)
print(f"b 的类型: {b.dtype}")  # 0 是整数

c = jnp.array(-jnp.inf)
print(f"c 的类型: {c.dtype}")  # -inf 是浮点数

d = jnp.array(jnp.nan)  # 补充测试：NaN
print(f"d 的类型: {d.dtype}")  # NaN 是浮点数

print(f"a 是否等于 b: {a == b}")
print(f"a 是否大于 c: {a > c}")
print(f"b 是否大于 c: {b > c}")

# NaN 比较测试
print(f"a ({a}) 是否大于 d ({d}): {a > d}")
print(f"a ({a}) 是否小于 d ({d}): {a < d}")
print(f"a ({a}) 是否等于 d ({d}): {a == d}")

# 补充的 NaN 与自身比较及 is_nan 测试
print(f"d ({d}) 是否等于 d ({d}): {d == d}")  # 预期为 False，NaN 不等于自身
print(f"是否是 NaN (jnp.isnan(d)): {jnp.isnan(d)}")  # 预期为 True，专门检查 NaN

# 补充的与 Python 原生数值的比较测试
print(f"b ({b}) 是否等于 Python int 0: {b == 0}")
print(f"a ({a}) 是否等于 Python int 0: {a == 0}")

# 补充的 jnp.isin 与 NaN 的交互测试
arr_with_nan = jnp.array([1, 2, 3, jnp.nan])
print(
    f"jnp.isin(d ({d}), arr_with_nan ({arr_with_nan})): {jnp.isin(d, arr_with_nan)}"
)  # 预期为 False
