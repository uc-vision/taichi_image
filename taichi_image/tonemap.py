import numpy as np

import math
import taichi as ti
import taichi.math as tm

@ti.func
def lerp(t, a, b):
  return a + t * (b - a)


@ti.func
def linear_func(image: ti.types.ndarray(dtype=tm.vec3, ndim=2), gamma:ti.f32):
    min = np.inf
    max = -np.inf

    for i, j in ti.ndrange(image.shape[0], image.shape[1]):
      for k in ti.static(range(3)):
        ti.atomic_min(min, image[i, j][k])
        ti.atomic_max(max, image[i, j][k])

    range = max - min

    for i, j in ti.ndrange(image.shape[0], image.shape[1]):
      image[i, j] = tm.pow((image[i, j] - min) / range, 1/gamma)


@ti.func
def rgb_gray(rgb) -> ti.f32:
  # 0.299⋅R+0.587⋅G+0.114⋅B
  return tm.dot(rgb, tm.vec3(0.299, 0.587, 0.114))

@ti.func
def bgr_gray(bgr) -> ti.f32:
  # 0.114⋅B+0.587⋅G+0.299⋅R
  return tm.dot(bgr, tm.vec3(0.114, 0.587, 0.299))

@ti.func
def image_statistics(image: ti.types.ndarray(dtype=tm.vec3, ndim=2)):
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
def reinhard_func(image: ti.template(), 
                    gamma:ti.f32, 
                    intensity:ti.f32, 
                    light_adapt:ti.f32, 
                    color_adapt:ti.f32):
   
  linear_func(image, 1.0)
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

    image[i, j] = image[i, j] * (1.0 / (adapt + image[i, j]))

  # Gamma correction
  linear_func(image, gamma)
  

@ti.kernel
def reinhard_kernel(image: ti.types.ndarray(dtype=tm.vec3, ndim=2), 
                    gamma:ti.f32, 
                    intensity:ti.f32, 
                    light_adapt:ti.f32, 
                    color_adapt:ti.f32):
  reinhard_func(image, gamma, intensity, light_adapt, color_adapt)

@ti.kernel
def linear_kernel(image: ti.types.ndarray(dtype=tm.vec3, ndim=2), gamma:ti.f32):
    linear_func(image, gamma)


def tonemap_linear(image, gamma=1.0):
  output = image.astype(np.float32).copy()  
  linear_kernel(output, gamma)
  return output


def tonemap_reinhard(image, gamma=1.0, intensity=1.0, light_adapt=1.0, color_adapt=0.0):
  output = image.astype(np.float32).copy()  
  reinhard_kernel(output, gamma, intensity, light_adapt, color_adapt)
  return output