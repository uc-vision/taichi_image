from typing import List, Optional, Tuple
from beartype import beartype
import taichi as ti
import taichi.math as tm
from taichi_image.types import empty_like, ti_to_torch
import torch

from taichi_image.util import Bounds, lerp, vec9, rgb_gray

from . import tonemap, interpolate, bayer, packed
import numpy as np

def moving_average(old, new, alpha):
  if old is None:
    return new
  
  return (1 - alpha) * old + alpha * new


# def expand_bounds(b:Bounds, x:ti.f32):
#   ti.atomic_min(b.min, x)
#   ti.atomic_max(b.max, x)


def camera_isp(name:str, dtype=ti.f32):
  decode12_kernel = packed.decode12_kernel(dtype, scaled=True)
  decode16_kernel = packed.decode16_kernel(dtype, scaled=True)

  torch_dtype = ti_to_torch[dtype]
  vec_dtype = ti.types.vector(3, dtype)


  @ti.kernel
  def load_u16(image: ti.types.ndarray(ti.u16, ndim=2),
               out:   ti.types.ndarray(dtype, ndim=2)):
    for i in ti.grouped(image):
      x = ti.cast(image[i], ti.f32) / 65535.0
      out[i] = ti.cast(x, dtype) 

  @ti.kernel
  def load_16f(image: ti.types.ndarray(ti.u16, ndim=2),
              out:   ti.types.ndarray(dtype, ndim=2)):
    for i in ti.grouped(image):
      out[i] = ti.cast(image[i], dtype)



  @ti.dataclass
  class Metering:
    bounds : Bounds
    log_bounds: Bounds
    log_mean: ti.f32
    mean: ti.f32
    rgb_mean: tm.vec3

    @ti.func
    def to_vec(self):
      return vec9(
        self.bounds.min, self.bounds.max,
        self.log_bounds.min, self.log_bounds.max,
                    self.log_mean, self.mean, *self.rgb_mean)

    @ti.func 
    def accum(self, rgb:tm.vec3, scaled:tm.vec3):
      gray = ti.f32(rgb_gray(scaled))
      log_gray = tm.log(tm.max(gray, 1e-4))

      ti.atomic_min(self.log_bounds.min, log_gray)
      ti.atomic_max(self.log_bounds.max, log_gray)

      for i in ti.static(range(3)):
        ti.atomic_min(self.bounds.min, rgb[i])
        ti.atomic_max(self.bounds.max, rgb[i])

      self.log_mean += log_gray
      self.mean += gray
      self.rgb_mean += scaled

    @ti.func
    def normalise(self, n:ti.i32):
      self.log_mean /= ti.f32(n)
      self.mean /= ti.f32(n)
      self.rgb_mean /= ti.f32(n)


  @ti.func
  def metering_from_vec(vec: ti.template()) -> Metering:
    return Metering(Bounds(vec[0], vec[1]), Bounds(vec[2], vec[3]), vec[4], vec[5], vec[6:9])

  @ti.kernel
  def metering_kernel(image: ti.types.ndarray(dtype=vec_dtype, ndim=2)) -> vec9:
                      
    bounds = Bounds(np.inf, -np.inf)
    for i, j in ti.ndrange(image.shape[0], image.shape[1]):
      for k in ti.static(range(3)):
        bounds.expand(image[i, j][k])


    stats = Metering(Bounds(np.inf, -np.inf), Bounds(np.inf, -np.inf), 0, 0, tm.vec3(0, 0, 0))      
    for i, j in ti.ndrange(image.shape[0], image.shape[1]):
      scaled = bounds.scale_range(image[i, j])
      stats.accum(image[i, j], scaled)


    stats.normalise(image.shape[0] * image.shape[1])
    return stats.to_vec()



  # @ti.kernel
  # def reinhard_kernel(image: ti.types.ndarray(dtype=vec_dtype, ndim=2), 
  #           dest: ti.types.ndarray(dtype=ti.types.vector(3, ti.u8), ndim=2),
  #           metering : vec9,
  #                     gamma:ti.f32, 
  #                     intensity:ti.f32, 
  #                     light_adapt:ti.f32, 
  #                     color_adapt:ti.f32) -> vec9:
    
  @ti.kernel
  def reinhard_kernel(image: ti.types.ndarray(dtype=vec_dtype, ndim=2), 
          dest: ti.types.ndarray(dtype=ti.types.vector(3, ti.u8), ndim=2),
          metering : vec9,
                    gamma:ti.template(), 
                    intensity:ti.template(),
                    light_adapt:ti.template(),
                    color_adapt:ti.template()) -> vec9:
    

    stats = metering_from_vec(metering)
    log_b = stats.log_bounds
    b = stats.bounds

    key = (log_b.max - stats.log_mean) / (log_b.max - log_b.min)
    map_key = 0.3 + 0.7 * tm.pow(key, 1.4)

    next_stats = Metering(Bounds(np.inf, -np.inf), Bounds(np.inf, -np.inf), 0, 0, tm.vec3(0, 0, 0))
    out_bounds = Bounds(np.inf, -np.inf)

    mean = lerp(color_adapt, stats.mean, stats.rgb_mean)
    for i in ti.grouped(ti.ndrange(image.shape[0], image.shape[1])):
  
      scaled =  (image[i] - b.min) / (b.max - b.min)
      next_stats.accum(image[i], scaled)

      gray = rgb_gray(scaled)
      
      # Blend between gray value and RGB value
      adapt_color = lerp(color_adapt, tm.vec3(gray), scaled)

      # Blend between mean and local adaptation
      adapt_mean = lerp(light_adapt, mean, adapt_color)
      adapt = tm.pow(tm.exp(-intensity) * adapt_mean, map_key)

      p = scaled * (1.0 / (adapt + scaled))
      for k in ti.static(range(3)):
        out_bounds.expand(p[k])

      image[i] = ti.cast(p, dtype)

    tonemap.linear_func(image, dest, out_bounds, gamma, 255, ti.u8)

    next_stats.normalise(image.shape[0] * image.shape[1])
    return next_stats.to_vec()



  @ti.data_oriented
  class ISP():
    @beartype
    def __init__(self, bayer_pattern:bayer.BayerPattern, 
                  scale:Optional[float]=None, resize_width:int=0,
                 moving_alpha=0.1, 
                 transform:interpolate.ImageTransform=interpolate.ImageTransform.none,
                 device:torch.device = torch.device('cuda', 0)):
      
      assert scale is None or resize_width == 0, "Cannot specify both scale and resize_width"    
  
      self.bayer_pattern = bayer_pattern
      self.moving_alpha = moving_alpha
      self.scale = scale
      self.resize_width = resize_width
      self.transform = transform

      self.metrics = None
      self.device = device

    @beartype
    def set(self, moving_alpha:Optional[float]=None, resize_width:Optional[int]=None, 
              scale:Optional[float]=None, 
              transform:Optional[interpolate.ImageTransform]=None):
      if moving_alpha is not None:
        self.moving_alpha = moving_alpha

      if resize_width is not None:
        self.resize_width = resize_width
        self.scale = None

      if scale is not None:
        self.scale = scale
        self.resize_width = 0

      if transform is not None:
        self.transform = transform
        


    def resize_image(self, image):
      w, h = image.shape[1], image.shape[0]
      if self.resize_width > 0:

        scale = self.resize_width / w 
        output_size = (self.resize_width, round(h * scale))
        return interpolate.resize_bilinear(image, output_size, scale, self.transform)
      elif self.scale is not None:

        output_size = (round(w * self.scale), round(h * self.scale))
        return interpolate.resize_bilinear(image, output_size, self.scale)
      elif self.transform != interpolate.ImageTransform.none:
        return interpolate.transform(image, self.transform)
      
      else:
        return image


    def load_16u(self, image):
      cfa = torch.empty(image.shape, dtype=torch_dtype, device=self.device)
      load_u16(image, cfa)
      return cfa

    def load_16f(self, image):
      cfa = torch.empty(image.shape, dtype=torch_dtype, device=self.device)
      load_16f(image, cfa)
      return cfa

    def load_packed12(self, image_data):
      w, h = (image_data.shape[1] * 2 // 3, image_data.shape[0])

      cfa = torch.empty(h, w, dtype=torch_dtype, device=self.device)    
      decode12_kernel(image_data.view(-1), cfa.view(-1))
      return cfa

    def load_packed16(self, image_data):
      w, h = (image_data.shape[1] // 2, image_data.shape[0])

      cfa = torch.empty(h, w, dtype=torch_dtype, device=self.device)    
      decode16_kernel(image_data.view(-1), cfa.view(-1))
      return cfa

    def updated_bounds(self, bounds:List[Bounds]):
      bounds = tonemap.union_bounds(bounds)
      self.moving_bounds = moving_average(self.moving_bounds, tonemap.bounds_to_np(bounds), self.moving_alpha)
      return self.moving_bounds
    
    def updated_metrics(self, image_metrics:List[vec9]):    
      mean_metrics = sum(image_metrics) / len(image_metrics)
      self.moving_metrics = moving_average(self.moving_metrics, mean_metrics, self.moving_alpha)
      return self.moving_metrics
        
    
    def tonemap_reinhard(self, cfa, gamma=1.0, intensity=1.0, light_adapt=1.0, color_adapt=0.0):
      rgb = bayer.bayer_to_rgb(cfa)
      image = self.resize_image(rgb) 

      output = torch.empty_like(image, dtype=torch.uint8, device=self.device) 

      if self.metrics is None:
        self.metrics = metering_kernel(image)

      self.metrics = reinhard_kernel(image, output, self.metrics, gamma, intensity, light_adapt, color_adapt)
      return output

  ISP.__qualname__ = name
  return ISP



Camera16 = camera_isp("Camera16", ti.f16)
Camera32 = camera_isp("Camera32", ti.f32)