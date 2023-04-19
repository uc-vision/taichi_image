from taichi_image.util import cache
from typing import List, Tuple
import numpy as np

import math
import taichi as ti
import taichi.math as tm

from taichi_image import types

@ti.func
def lerp(t, a, b):
  return a + t * (b - a)

@ti.func
def bounds_from_vec(v:tm.vec2):
  return Bounds(v[0], v[1])

@ti.dataclass
class Bounds:
  min: ti.f32
  max: ti.f32

  @ti.func
  def span(self):
    return self.max - self.min

  @ti.func
  def union(self, other):
    self.min=ti.min(self.min, other.min)
    self.max=ti.max(self.max, other.max)

  @ti.func
  def to_vec(self):
    return tm.vec2(self.min, self.max)
  
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
def linear_func(image: ti.template(), output:ti.template(), bounds:Bounds, gamma:ti.f32, scale_factor:ti.f32, dtype):
    inv_range = 1 / (bounds.max - bounds.min)

    for i in ti.grouped(ti.ndrange(*image.shape)):
      x = tm.pow((image[i] - bounds.min) * inv_range, 1/gamma)
      output[i] = ti.cast(tm.clamp(x, 0, 1) * scale_factor, dtype)

@ti.func
def rescale_func(image: ti.template(), bounds:Bounds, dtype):
    inv_range = 1 / (bounds.max - bounds.min)
    for i in ti.grouped(ti.ndrange(*image.shape)):
      image[i] = ti.cast((image[i] - bounds.min) * inv_range, dtype)

@ti.kernel 
def rescale_kernel(image: ti.types.ndarray(ndim=3), bounds:Bounds, dtype:ti.i32):
  rescale_func(image, bounds, dtype)


@ti.func
def gamma_func(image: ti.template(), output:ti.template(), gamma:ti.f32, scale_factor:ti.f32, dtype):

    for i in ti.grouped(ti.ndrange(*image.shape)):
      x = tm.pow(image[i], 1/gamma)
      output[i] = ti.cast(tm.clamp(x, 0, 1) * scale_factor, dtype)      

@ti.kernel
def linear_kernel(src: ti.types.ndarray(ndim=3), dest: ti.types.ndarray(ndim=3), 
                  gamma:ti.f32, scale_factor:ti.f32, dtype:ti.template()):
    linear_func(src, dest, bounds_func(src), gamma, scale_factor, dtype)



def tonemap_linear(src, gamma=1.0, dtype=None, scale_factor=1.0):
  output = types.empty_like(src, src.shape, dtype or types.ti_type[src])
  linear_kernel(src, output, gamma, scale_factor)
  return output


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


vec7 = ti.types.vector(7, ti.f32)

@ti.func
def metering_from_vec(vec: ti.template()):
  return Metering(Bounds(vec[0], vec[1]), vec[2], vec[3], vec[4:7])

@ti.dataclass
class Metering:
  log_bounds: Bounds
  log_mean: ti.f32
  gray_mean: ti.f32
  rgb_mean: tm.vec3

  @ti.func
  def to_vec(self):
    return vec7(self.log_bounds.min, self.log_bounds.max, 
                   self.log_mean, self.gray_mean, *self.rgb_mean)

  
def metering_to_np(x:Metering):
  return np.array([x.log_bounds.min, x.log_bounds.max, 
                   x.log_mean, x.gray_mean, *x.rgb_mean])

def metering_from_np(x:np.ndarray):
  return Metering(Bounds(x[0], x[1]), x[2], x[3], 
                  tm.vec3(x[4], x[5], x[6]))




@ti.func
def metering_func(image: ti.template(), bounds:Bounds) -> Metering:
  total_log_gray = 0.0
  total_gray = 0.0
  total_rgb = tm.vec3(0.0)
  
  log_min = ti.f32(np.inf)
  log_max = ti.f32(np.inf)

  for i, j in ti.ndrange(image.shape[0], image.shape[1]):
    scaled = (image[i, j] - bounds.min) / (bounds.max - bounds.min)

    gray = ti.f32(rgb_gray(scaled))
    log_gray = tm.log(tm.max(gray, 1e-4))

    # To side-step a bug use negative atomic_min instead of atomic_max
    ti.atomic_min(log_max, -log_gray)
    ti.atomic_min(log_min, log_gray)

    total_log_gray += log_gray
    total_gray += gray
    total_rgb += image[i, j]

  n = (image.shape[0] * image.shape[1])
  return Metering(Bounds(log_min, -log_max), 
                        total_log_gray / n, total_gray / n, total_rgb / n)


@ti.kernel 
def metering_kernel(image: ti.types.ndarray(ndim=2)) -> Metering:
  return metering_func(image)

@ti.func
def reinhard_func(image : ti.template(),
                  bounds : Bounds,
                  stats : Metering,
                    intensity:ti.f32, 
                    light_adapt:ti.f32, 
                    color_adapt:ti.f32,
                    dtype:ti.template()):
  
  b = stats.log_bounds
  key = (b.max - stats.log_mean) / (b.max - b.min)
  map_key = 0.3 + 0.7 * tm.pow(key, 1.4)

  mean = lerp(color_adapt, stats.gray_mean, stats.rgb_mean)
  for i, j in ti.ndrange(image.shape[0], image.shape[1]):
    scaled = (image[i, j] - bounds.min) / (bounds.max - bounds.min)
    gray = rgb_gray(scaled)

    # Blend between gray value and RGB value
    adapt_color = lerp(color_adapt, tm.vec3(gray), scaled)

    # Blend between mean and local adaptation
    adapt_mean = lerp(light_adapt, mean, adapt_color)
    adapt = tm.pow(tm.exp(-intensity) * adapt_mean, map_key)

    image[i, j] = ti.cast(scaled * (1.0 / (adapt + scaled)), dtype) 


@cache
def reinhard_kernel(in_dtype=ti.f32, out_dtype=ti.f32):

  @ti.kernel
  def k(image: ti.types.ndarray(dtype=ti.types.vector(3, in_dtype), ndim=2), 
             dest: ti.types.ndarray(dtype=ti.types.vector(3, out_dtype), ndim=2),
                      gamma:ti.f32, 
                      intensity:ti.f32, 
                      light_adapt:ti.f32, 
                      color_adapt:ti.f32):
    
    bounds = bounds_func(image)
    linear_func(image, dest, bounds, gamma, 1.0, out_dtype)

    stats = metering_func(image, bounds)
    reinhard_func(image, bounds, stats, intensity, light_adapt, color_adapt, in_dtype)

    # Gamma correction
    min, max = bounds_func(image)
    linear_func(image, dest, min, max, gamma, types.scale_factor[out_dtype], out_dtype)
  return k




def tonemap_reinhard(src, gamma=1.0, intensity=1.0, light_adapt=1.0, color_adapt=0.0, dtype=ti.uint8):

  output = types.empty_like(src, src.shape, dtype)
  temp = types.empty_like(src, src.shape, ti.f32)

  tonemap_kernel = reinhard_kernel(types.ti_type(src), dtype)
  tonemap_kernel(src, temp, output, 
                gamma, intensity, light_adapt, color_adapt)
  return output





