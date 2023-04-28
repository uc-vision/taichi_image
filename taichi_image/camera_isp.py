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


def image_bounds(image):
  return torch.concatenate([image.min().view(1), image.max().view(1)])

def metering(image, bounds):
  weights = torch.tensor([0.299, 0.587, 0.114], dtype=image.dtype, device=image.device)
  
  image = (image - bounds[0]) / (bounds[1] - bounds[0])
  grey = torch.einsum('ijk,k -> ij', image, weights)
  grey_mean = grey.mean()

  log_grey = torch.log(torch.clamp(grey, min=1e-4))

  return torch.concatenate([
    log_grey.min().view(1), log_grey.max().view(1),
                log_grey.mean().view(1), grey_mean.view(1), image.mean(dim=(0, 1))], dim=0)

def metering_images(images, t, prev):
    images_bounds = torch.stack([image_bounds(image) for image in images])
    bounds = torch.concatenate([images_bounds[:, 0].min().view(1), 
                                images_bounds[:, 1].max().view(1)])

    new_bounds = t * prev[:2] + (1.0 - t) * bounds

    image_stats = torch.stack([metering(image, new_bounds) for image in images])
    stats = image_stats.mean(dim=0)

    new_stats = t * prev[2:] + (1.0 - t) * stats
    return torch.concatenate([new_bounds, new_stats])


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
        self.log_bounds.min, self.log_bounds.max,
                    self.log_mean, self.mean, *self.rgb_mean)
  
  def linear_output(image, gamma=1.0):
    upper = torch.max(image)
    linear = 255 * (image / upper).pow(1/gamma)
    return linear.to(torch.uint8)


  @ti.func
  def metering_from_vec(vec: ti.template()) -> Metering:
    return Metering(Bounds(vec[0], vec[1]), Bounds(vec[2], vec[3]), vec[4], vec[5], tm.vec3(vec[6], vec[7], vec[8]))


    
  @ti.kernel
  def reinhard_kernel(image: ti.types.ndarray(dtype=vec_dtype, ndim=2), 
          metering : ti.types.ndarray(dtype=ti.f32, ndim=1),
                    intensity:ti.template(),
                    light_adapt:ti.template(),
                    color_adapt:ti.template()):
    

    stats = metering_from_vec(metering)
    log_b = stats.log_bounds
    b = stats.bounds

    key = (log_b.max - stats.log_mean) / (log_b.max - log_b.min)
    map_key = 0.3 + 0.7 * tm.pow(key, 1.4)

    mean = lerp(color_adapt, stats.mean, stats.rgb_mean)
    for i in ti.grouped(ti.ndrange(image.shape[0], image.shape[1])):
  
      scaled =  (image[i] - b.min) / (b.max - b.min)
      gray = rgb_gray(scaled)
      
      # Blend between gray value and RGB value
      adapt_color = lerp(color_adapt, tm.vec3(gray), scaled)

      # Blend between mean and local adaptation
      adapt_mean = lerp(light_adapt, mean, adapt_color)
      adapt = tm.pow(tm.exp(-intensity) * adapt_mean, map_key)

      p = scaled * (1.0 / (adapt + scaled))
      image[i] = ti.cast(p, dtype)



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
        
    def _process_image(self, cfa):
      rgb = bayer.bayer_to_rgb(cfa)
      return self.resize_image(rgb) 



    @torch.compile(backend="cudagraphs")
    def tonemap_reinhard(self, cfa_images:List[torch.Tensor], 
                         gamma:float=1.0, intensity:float=1.0, light_adapt:float=1.0, color_adapt:float=0.0):
      images = [self._process_image(cfa) for cfa in cfa_images]

      if self.metrics is None:
        self.metrics = metering_images(images, 0.0, torch.zeros(3, dtype=torch_dtype, device=self.device))
      else:
        self.metrics = metering_images(images, self.moving_alpha, self.metrics)

      for image in images:
        reinhard_kernel(image, self.metrics, intensity, light_adapt, color_adapt)
      
      return [linear_output(image, gamma) for image in images]
    


  ISP.__qualname__ = name
  return ISP



Camera16 = camera_isp("Camera16", ti.f16)
Camera32 = camera_isp("Camera32", ti.f32)