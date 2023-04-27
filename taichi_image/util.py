from functools import lru_cache
from typing import List
import taichi as ti
import taichi.math as tm
import numpy as np

cache = lru_cache(maxsize=None)


vec5 = ti.types.vector(5, ti.f32)
vec6 = ti.types.vector(6, ti.f32)
vec7 = ti.types.vector(7, ti.f32)
vec8 = ti.types.vector(8, ti.f32)
vec9 = ti.types.vector(9, ti.f32)
vec10 = ti.types.vector(10, ti.f32)
vec11 = ti.types.vector(11, ti.f32)
vec12 = ti.types.vector(12, ti.f32)



@ti.dataclass
class Bounds:
  min: ti.f32
  max: ti.f32

  @ti.func
  def span(self):
    return self.max - self.min

  @ti.func
  def union(self, other):
    ti.atomic_min(self.min, other.min)
    ti.atomic_min(-self.max, -other.max)

  @ti.func
  def expand(self, v):
    ti.atomic_min(self.min, v)
    ti.atomic_min(-self.max, -v)


  @ti.func
  def to_vec(self):
    return tm.vec2(self.min, self.max)
  

  @ti.func
  def scale_range(self, v:ti.template()):
    return (v - self.min) / self.span
  
@ti.func 
def bounds_func(image: ti.template()) -> Bounds:
    min = np.inf
    max = -np.inf

    for i in ti.grouped(ti.ndrange(*image.shape[:2])):
      for k in ti.static(range(3)):
        ti.atomic_min(min, ti.cast(image[i][k], ti.f32))
        ti.atomic_max(max, ti.cast(image[i][k], ti.f32))


    return Bounds(min, max)


def union_bounds(bounds:List[Bounds]):
  result = Bounds(np.inf, -np.inf)
  for b in bounds:
    result.min = min(result.min, b.min)
    result.max = max(result.max, b.max)

  return result

def bounds_to_np(b:Bounds):
  return np.array([b.min, b.max])

def bounds_from_np(b:np.ndarray):
  return Bounds(b[0], b[1])



@ti.func
def rgb_gray(rgb) -> ti.f32:
  # 0.299⋅R+0.587⋅G+0.114⋅B
  return tm.dot(rgb, tm.vec3(0.299, 0.587, 0.114))

@ti.func
def bgr_gray(bgr) -> ti.f32:
  # 0.114⋅B+0.587⋅G+0.299⋅R
  return tm.dot(bgr, tm.vec3(0.114, 0.587, 0.299))

def rgb_linear(rgb):
  return ti.select(rgb <= 0.04045, 
     rgb / 12.92,
    tm.pow((rgb + 0.055) / 1.055, 2.4))

@ti.func
def rgb_ciexyz(rgb:tm.vec3):
  linear = rgb_linear(rgb)
  m = tm.mat3x3(
    0.4124564, 0.3575761, 0.1804375,
    0.2126729, 0.7151522, 0.0721750,
    0.0193339, 0.1191920, 0.9503041
  )
  return m @ linear


@ti.func
def lerp(t, a, b):
  return a + t * (b - a)
