from typing import List, Optional, Tuple
import taichi as ti
import taichi.math as tm
from taichi_image.types import empty_like, ti_to_torch
import torch

from . import tonemap, interpolate, bayer, packed
import numpy as np

def moving_average_with(to_vec, from_vec,
    old, new, alpha):
  
  if old is None:
    return new
  
  return from_vec(alpha * to_vec(old) + (1 - alpha) * to_vec(new))


def camera_isp(dtype=ti.f32, device:torch.device = torch.device('cuda:0')):
  decode12_kernel = packed.decode12_kernel(dtype)
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

  @ti.kernel 
  def  bounds_kernel(image: ti.types.ndarray(dtype=vec_dtype, ndim=2)) -> tonemap.Bounds:
    return tonemap.bounds_func(image)

  @ti.kernel
  def linear_kernel(image: ti.types.ndarray(dtype=ti.types.vector(3, ti.u8), ndim=2),
                    output:ti.types.ndarray(dtype=vec_dtype, ndim=2),
                    bounds:tonemap.Bounds, gamma:ti.f32):
    tonemap.linear_func(image, output, bounds, gamma, 255, dtype)


  @ti.data_oriented
  class ISP():
    def __init__(self, bayer_pattern:bayer.BayerPattern, 
                  scale:Optional[float]=None, resize_width:int=0,
                 moving_alpha=0.1):
      assert scale is None or resize_width == 0, "Cannot specify both scale and resize_width"    
  
      self.bayer_pattern = bayer_pattern
      self.moving_alpha = moving_alpha
      self.scale = scale
      self.resize_width = resize_width

      self.moving_bounds = None
      self.moving_metrics = None


    def resize_image(self, image):
      w, h = image.shape[1], image.shape[0]
      if self.scale is not None:

        output_size = (round(w * self.scale), round(h * self.scale))
        return interpolate.resize_bilinear(image, output_size, self.scale)

      elif self.resize_width > 0:

        scale = self.resize_width / w 
        output_size = (self.resize_width, round(h * scale))
        return interpolate.resize_bilinear(image, output_size, scale)

      else:
        return image


    def load_16u(self, image):
      cfa = torch.empty(image.shape, dtype=torch_dtype, device=device)
      load_u16(image, cfa)
      return self._process_image(cfa)

    def load_16f(self, image):
      cfa = torch.empty(image.shape, dtype=torch_dtype, device=device)
      load_16f(image, cfa)
      return self._process_image(cfa)

    def load_packed12(self, image_data, image_size):
      cfa = torch.empty(image_size[1], image_size[0], dtype=torch_dtype, device=device)
      decode12_kernel(image_data, cfa.view(-1))
      return self._process_image(cfa)

    def _process_image(self, cfa):
      rgb = bayer.bayer_to_rgb(cfa)
      return self.resize_image(rgb)

    def updated_bounds(self, images):
      bounds = tonemap.union_bounds([bounds_kernel(image) for image in images])
      return moving_average_with(tonemap.bounds_to_np, tonemap.bounds_from_np,
                             self.moving_bounds, bounds, self.moving_alpha)

    
    def updated_metrics(self, images):
      image_metrics = [tonemap.metering_to_np(tonemap.metering_kernel(image, self.moving_bounds)) 
                 for image in images]
      mean_metrics = tonemap.metering_from_np(np.mean(image_metrics, axis=0))
      metrics = moving_average_with(tonemap.metering_to_np, tonemap.metering_from_np,
                             mean_metrics, self.moving_metrics, self.moving_alpha)

      return metrics
        


    def tonemap_linear(self, images, gamma=1.0):

      outputs = [torch.empty_like(image, dtype=torch.uint8, device=device) for image in images]
      self.moving_bounds = self.updated_bounds(images)

      return [linear_kernel(image, output, gamma) 
              for image, output in zip(images, outputs)]
  
    def tonemap_reinhard(self, images, gamma=1.0, intensity=1.0, light_adapt=1.0, color_adapt=0.0):
      outputs = [torch.empty_like(image, dtype=torch.uint8, device=device) for image in images]

      self.moving_bounds = self.updated_bounds(images)
      self.moving_metrics = self.updated_metrics(images)

      return [reinhard_kernel(image, output, self.moving_bounds, self.moving_metrics, gamma, intensity, light_adapt, color_adapt) 
              for image, output in zip(images, outputs)]
          
  return ISP



