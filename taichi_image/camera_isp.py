from typing import List, Optional, Tuple
import taichi as ti
import taichi.math as tm
from taichi_image.types import empty_like, ti_to_torch
import torch

from . import tonemap, interpolate, bayer, packed
import numpy as np
from py_structs.torch import shape_info

def moving_average(old, new, alpha):
  if old is None:
    return new
  
  return (1 - alpha) * old + alpha * new



def camera_isp(name:str, dtype=ti.f32):
  decode12_kernel = packed.decode12_kernel(dtype, scaled=True)
  decode16_kernel = packed.decode16_kernel(dtype, scaled=True)

  torch_dtype = ti_to_torch[dtype]
  vec_dtype = ti.types.vector(3, dtype)
  vec7 = ti.types.vector(7, ti.f32)


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

  @ti.kernel 
  def  bounds_kernel(image: ti.types.ndarray(dtype=vec_dtype, ndim=2)) -> tonemap.Bounds:
    return tonemap.bounds_func(image)

  @ti.kernel
  def linear_kernel(image: ti.types.ndarray(dtype=ti.types.vector(3, ti.u8), ndim=2),
                    output:ti.types.ndarray(dtype=vec_dtype, ndim=2),
                    bounds:tm.vec2, gamma:ti.f32):
    tonemap.linear_func(image, output,tonemap.bounds_from_vec(bounds), gamma, 255, ti.u8)

  # @ti.kernel
  # def reinhard_kernel(image: ti.types.ndarray(dtype=vec_dtype, ndim=2),
  #                     output:ti.types.ndarray(dtype=ti.types.vector(3, ti.u8), ndim=2),
  #                     bounds:tm.vec2,
  #                     metrics:vec7,
  #                     gamma:ti.f32, intensity:ti.f32,
  #                     light_adapt:ti.f32, color_adapt:ti.f32):
    
  @ti.kernel
  def reinhard_kernel(image: ti.types.ndarray(dtype=vec_dtype, ndim=2),
                      output:ti.types.ndarray(dtype=ti.types.vector(3, ti.u8), ndim=2),
                      bounds:tm.vec2,
                      metrics:vec7,
                      gamma:ti.template(), intensity:ti.template(),
                      light_adapt:ti.template(), color_adapt:ti.template()):
    tonemap.reinhard_func(image, tonemap.bounds_from_vec(bounds), tonemap.metering_from_vec(metrics), 
                          intensity, light_adapt, color_adapt, dtype)
    
    after_bounds = tonemap.bounds_func(image)
    tonemap.linear_func(image, output, after_bounds, gamma, 255, ti.u8)


  @ti.kernel
  def metering_kernel(image: ti.types.ndarray(dtype=vec_dtype, ndim=2), bounds:tm.vec2) -> tonemap.Metering:
    return tonemap.metering_func(image, tonemap.bounds_from_vec(bounds))


  @ti.data_oriented
  class ISP():
    def __init__(self, bayer_pattern:bayer.BayerPattern, 
                  scale:Optional[float]=None, resize_width:int=0,
                 moving_alpha=0.1, 
                 transform:interpolate.ImageTransform=interpolate.ImageTransform.none,
                 device:torch.device = torch.device('cuda:0')):
      assert scale is None or resize_width == 0, "Cannot specify both scale and resize_width"    
  
      self.bayer_pattern = bayer_pattern
      self.moving_alpha = moving_alpha
      self.scale = scale
      self.resize_width = resize_width
      self.transform = transform

      self.moving_bounds = None
      self.moving_metrics = None
      self.device = device

    def set(self, moving_alpha=None, resize_width=None, scale=None, transform=None):
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
      if self.scale is not None:

        output_size = (round(w * self.scale), round(h * self.scale))
        return interpolate.resize_bilinear(image, output_size, self.scale)

      elif self.resize_width > 0 or self.transform != interpolate.ImageTransform.none:

        scale = self.resize_width / w 
        output_size = (self.resize_width, round(h * scale))
        return interpolate.resize_bilinear(image, output_size, scale, self.transform)

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

    def updated_bounds(self, images):
      bounds = tonemap.union_bounds([bounds_kernel(image) for image in images])

      self.moving_bounds = moving_average(self.moving_bounds, tonemap.bounds_to_np(bounds), self.moving_alpha)
      return self.moving_bounds
    
    def updated_metrics(self, images):
      image_metrics = [tonemap.metering_to_np(metering_kernel(image, self.moving_bounds)) 
                 for image in images]
      
      mean_metrics = np.mean(image_metrics, axis=0)
      self.moving_metrics = moving_average(self.moving_metrics, mean_metrics, self.moving_alpha)
      return self.moving_metrics
        

    def _process_input(self, cfa):
      rgb = bayer.bayer_to_rgb(cfa)
      return self.resize_image(rgb)

    def tonemap_linear(self, images, gamma=1.0):
      images = [self._process_input(image) for image in images]

      outputs = [torch.empty_like(image, dtype=torch.uint8, device=self.device) for image in images]
      self.moving_bounds = self.updated_bounds(images)

      for image, output in zip(images, outputs):
        linear_kernel(image, output, tm.vec2(*self.moving_bounds), gamma)
      
      return outputs
    
    def tonemap_reinhard(self, images, gamma=1.0, intensity=1.0, light_adapt=1.0, color_adapt=0.0):
      images = [self._process_input(image) for image in images]

      self.moving_bounds = self.updated_bounds(images)
      self.moving_metrics = self.updated_metrics(images)

      outputs = [torch.empty_like(image, dtype=torch.uint8, device=self.device) for image in images]
      for image, output in zip(images, outputs):
        reinhard_kernel(image, output, tm.vec2(*self.moving_bounds), vec7(*self.moving_metrics), 
                        gamma, intensity, light_adapt, color_adapt)
        
      del images
      return outputs

  ISP.__qualname__ = name
  return ISP



Camera16 = camera_isp("Camera16", ti.f16)
Camera32 = camera_isp("Camera32", ti.f32)