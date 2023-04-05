from taichi_image.util import cache
from typing import Tuple
import numpy as np

import math
import taichi as ti
import taichi.math as tm

from taichi_image import types

@ti.func
def lerp(t, a, b):
  return a + t * (b - a)

@ti.func 
def min_max(image: ti.template()) -> Tuple[float, float]:
    min = np.inf
    max = -np.inf

    for i in ti.grouped(ti.ndrange(*image.shape[:2])):
      for k in ti.static(range(3)):
        ti.atomic_min(min, ti.cast(image[i][k], ti.f32))
        ti.atomic_max(max, ti.cast(image[i][k], ti.f32))

    return min, max


@ti.func
def linear_func(image: ti.template(), output:ti.template(), gamma:ti.f32, scale_factor:ti.f32, dtype):
    min, max = min_max(image)
    range = max - min

    for i in ti.grouped(ti.ndrange(*image.shape)):
      output[i] = ti.cast(tm.pow((image[i] - min) / range, 1/gamma) * scale_factor, dtype)


@ti.func
def normalise_range(image: ti.template(), dest: ti.template()):
    min, max = min_max(image)
    range = max - min

    for i in ti.grouped(ti.ndrange(*image.shape)):
      dest[i] = (image[i] - min) / range

@ti.kernel
def linear_kernel(src: ti.types.ndarray(ndim=3), dest: ti.types.ndarray(ndim=3), gamma:ti.f32, scale_factor:ti.f32):
    linear_func(src, dest, gamma, scale_factor, dest.dtype)


def tonemap_linear(src, gamma=1.0, dtype=None, scale_factor=1.0):
  output = types.empty_array(src, src.shape, dtype or types.ti_type[src])
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


@cache
def reinhard_kernel(in_dtype=ti.f32, out_dtype=ti.f32):

  @ti.func
  def image_statistics(image: ti.template()):
    total_log_gray = 0.0
    total_gray = 0.0
    total_rgb = tm.vec3(0.0)
    
    log_min = ti.f32(np.inf)
    log_max = ti.f32(np.inf)

    for i, j in ti.ndrange(image.shape[0], image.shape[1]):
      gray = ti.f32(rgb_gray(image[i, j]))
      log_gray = tm.log(tm.max(gray, 1e-4))

      # To side-step a bug use negative atomic_min instead of atomic_max
      ti.atomic_min(log_max, -log_gray)
      ti.atomic_min(log_min, log_gray)

      total_log_gray += log_gray
      total_gray += gray
      total_rgb += image[i, j]

    n = (image.shape[0] * image.shape[1])
    log_mean = total_log_gray / n
    gray_mean = total_gray / n
    image_mean = total_rgb / n


    return log_min, -log_max, log_mean, gray_mean, image_mean



  @ti.func
  def reinhard_func(image : ti.template(),
                    output: ti.template(),
                      gamma:ti.f32, 
                      intensity:ti.f32, 
                      light_adapt:ti.f32, 
                      color_adapt:ti.f32):
    
  
    log_min, log_max, log_mean, gray_mean, image_mean = image_statistics(image)

    key = (log_max - log_mean) / (log_max - log_min)
    map_key = 0.3 + 0.7 * tm.pow(key, 1.4)

    mean = lerp(color_adapt, gray_mean, image_mean)
    for i, j in ti.ndrange(image.shape[0], image.shape[1]):
      gray = rgb_gray(image[i, j])

      # Blend between gray value and RGB value
      adapt_color = lerp(color_adapt, tm.vec3(gray), image[i, j])

      # Blend between mean and local adaptation
      adapt_mean = lerp(light_adapt, mean, adapt_color)
      adapt = tm.pow(tm.exp(-intensity) * adapt_mean, map_key)

      image[i, j] = (image[i, j] * (1.0 / (adapt + image[i, j]))) 


    # Gamma correction
    linear_func(image, output, gamma, types.scale_factor[out_dtype], out_dtype)
    

  @ti.kernel
  def kernel(src: ti.types.ndarray(dtype=ti.types.vector(3, in_dtype), ndim=2), 
             image: ti.types.ndarray(dtype=ti.types.vector(3, ti.f32), ndim=2), 
             dest: ti.types.ndarray(dtype=ti.types.vector(3, out_dtype), ndim=2), 
                      gamma:ti.f32, 
                      intensity:ti.f32, 
                      light_adapt:ti.f32, 
                      color_adapt:ti.f32):
    
    normalise_range(src, image)
    reinhard_func(image, dest, gamma, intensity, light_adapt, color_adapt)

  return kernel

def tonemap_reinhard(src, gamma=1.0, intensity=1.0, light_adapt=1.0, color_adapt=0.0, dtype=ti.uint8):

  output = types.empty_array(src, src.shape, dtype)
  temp = types.empty_array(src, src.shape, ti.f32)

  kernel = reinhard_kernel(types.ti_type(src), dtype)

  kernel(src, temp, output, gamma, intensity, light_adapt, color_adapt)
  return output