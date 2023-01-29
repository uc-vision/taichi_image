import taichi as ti
from taichi.math import vec2, vec3, vec4, ivec2, ivec3, ivec4

u8vec3 = ti.types.vector(3, ti.u8)
u16vec3 = ti.types.vector(3, ti.u16)
f16vec3 = ti.types.vector(3, ti.f16)


def scale_factor(dtype):
  if dtype == ti.u8:
    return 255
  elif dtype == ti.u16:
    return 65535
  elif dtype == ti.f32:
    return 1.0
  else:
    raise ValueError(f"unsupported dtype {dtype}")