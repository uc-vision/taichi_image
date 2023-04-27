from taichi_image.util import Bounds, bounds_func, cache, vec7, rgb_gray, lerp
import numpy as np

import taichi as ti
import taichi.math as tm

from taichi_image import types


@ti.func
def linear_func(image: ti.template(), output:ti.template(), bounds:Bounds, gamma:ti.f32, scale_factor:ti.f32, dtype):
    inv_range = 1 / (bounds.max - bounds.min)

    for i in ti.grouped(ti.ndrange(*image.shape)):
      x = tm.pow((image[i] - bounds.min) * inv_range, 1/gamma)
      output[i] = ti.cast(tm.clamp(x, 0, 1) * scale_factor, dtype)


@ti.func
def gamma_func(image: ti.template(), output:ti.template(), gamma:ti.f32, scale_factor:ti.f32, dtype):
    for i in ti.grouped(ti.ndrange(*image.shape)):
      x = tm.pow(image[i], 1/gamma)
      output[i] = ti.cast(tm.clamp(x, 0, 1) * scale_factor, dtype)      

@cache
def linear_kernel(in_dtype, out_dtype):
  in_vec = ti.types.vector(3, in_dtype)
  out_vec = ti.types.vector(3, out_dtype)

  @ti.kernel
  def k(src: ti.types.ndarray(in_vec, ndim=2), dest: ti.types.ndarray(out_vec, ndim=2), 
                  gamma:ti.f32, scale_factor:ti.f32):
    
    bounds = bounds_func(src)
    print("bounds", bounds.min, bounds.max)
    linear_func(src, dest, bounds, gamma, scale_factor, out_dtype)

  return k


def tonemap_linear(src, gamma=1.0, dtype=ti.u8):
  output = types.empty_like(src, src.shape, dtype)
  k = linear_kernel(types.ti_type(src), dtype)

  k(src, output, gamma, types.scale_factor[dtype])
  return output


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
  mean_rgb = total_rgb / n

  return Metering(Bounds(log_min, -log_max), 
                        total_log_gray / n, total_gray / n, mean_rgb)



@ti.func
def reinhard_func(image : ti.template(),
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
    scaled = image[i, j] 
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
        temp : ti.types.ndarray(dtype=ti.types.vector(3, ti.f32), ndim=2),
             dest: ti.types.ndarray(dtype=ti.types.vector(3, out_dtype), ndim=2),
                      gamma:ti.f32, 
                      intensity:ti.f32, 
                      light_adapt:ti.f32, 
                      color_adapt:ti.f32):
    
    bounds = bounds_func(image)
    linear_func(image, temp, bounds, gamma, 1.0, ti.f32)

    stats = metering_func(temp, Bounds(0, 1))
    reinhard_func(temp, stats, intensity, light_adapt, color_adapt, ti.f32)

    # Gamma correction
    bounds2 = bounds_func(temp)
    linear_func(temp, dest, bounds2, gamma, types.scale_factor[out_dtype], out_dtype)
  return k




def tonemap_reinhard(src, gamma=1.0, intensity=1.0, light_adapt=1.0, color_adapt=0.0, dtype=ti.uint8):

  output = types.empty_like(src, src.shape, dtype)
  temp = types.empty_like(src, src.shape, ti.f32)

  tonemap_kernel = reinhard_kernel(types.ti_type(src), dtype)
  tonemap_kernel(src, temp, output, 
                gamma, intensity, light_adapt, color_adapt)
  return output





