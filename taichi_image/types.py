import taichi as ti
from taichi.math import vec2, vec3, vec4, ivec2, ivec3, ivec4
import numpy as np

u8vec3 = ti.types.vector(3, ti.u8)
u16vec3 = ti.types.vector(3, ti.u16)
f16vec3 = ti.types.vector(3, ti.f16)



pixel_types = {
  'u8': (ti.u8, 255),
  'u16': (ti.u16, 65535),
  'f16': (ti.f16, 1.0),
  'f32': (ti.f32, 1.0)
}


ti_to_np = {
  ti.u8: np.uint8,
  ti.u16: np.uint16,
  ti.f16: np.float16,
  ti.f32: np.float32
}

np_to_ti = {
  np.uint8: ti.u8,
  np.uint16: ti.u16,
  np.float16: ti.f16,
  np.float32: ti.f32
}
