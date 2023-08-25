import taichi as ti
import taichi.math as tm

from taichi_image.color.yuv_420 import *


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
  m = tm.mat3(
    0.4124564, 0.3575761, 0.1804375,
    0.2126729, 0.7151522, 0.0721750,
    0.0193339, 0.1191920, 0.9503041
  )
  return m @ linear

