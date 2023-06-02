from typing import List, Optional, Tuple
from beartype import beartype
import taichi as ti
import taichi.math as tm
from taichi_image.types import empty_like, ti_to_torch
import torch
import torch.nn.functional as F

from taichi_image.util import Bounds, lerp, vec9, rgb_gray

from . import tonemap, interpolate, bayer, packed
import numpy as np

def moving_average(old, new, alpha):
  if old is None:
    return new
  
  return (1 - alpha) * old + alpha * new


def image_bounds(image):
  return torch.concatenate([image.min().view(1), image.max().view(1)])

def metering_torch(image, bounds):
  weights = torch.tensor([0.299, 0.587, 0.114], dtype=image.dtype, device=image.device)
  
  image = (image - bounds[0]) / (bounds[1] - bounds[0])
  grey = torch.einsum('ijk,k -> ij', image, weights)
  grey_mean = grey.mean()

  log_grey = torch.log(torch.clamp(grey, min=1e-4))

  return torch.concatenate([
    log_grey.min().view(1), log_grey.max().view(1),
                log_grey.mean().view(1), grey_mean.view(1), image.mean(dim=(0, 1))], dim=0)

@torch.compile
def metering_images_torch(images, t, prev, stride=8):
    images = torch.concatenate([image[::stride, ::stride, :] for image in images], 0)
    bounds = image_bounds(images)

    new_bounds = t * prev[:2] + (1.0 - t) * bounds

    stats = metering_torch(images, new_bounds)
    new_stats = t * prev[2:] + (1.0 - t) * stats

    return torch.concatenate([new_bounds, new_stats])




def transform(image:torch.Tensor, transform:interpolate.ImageTransform):
  if transform == interpolate.ImageTransform.none:
    return image
  elif transform == interpolate.ImageTransform.rotate_90:
    return torch.rot90(image, 1, (0, 1)).contiguous()
  elif transform == interpolate.ImageTransform.rotate_180:
    return torch.rot90(image, 2, (0, 1)).contiguous()
  elif transform == interpolate.ImageTransform.rotate_270:
    return torch.rot90(image, 3, (0, 1)).contiguous()
  elif transform == interpolate.ImageTransform.flip_horiz:
    return torch.flip(image, (1,)).contiguous()
  elif transform == interpolate.ImageTransform.flip_vert:
    return torch.flip(image, (0,)).contiguous()
  elif transform == interpolate.ImageTransform.transverse:
    return torch.flip(image, (0, 1)).contiguous()
  elif transform == interpolate.ImageTransform.transpose:
    return torch.transpose(image, 0, 1).contiguous()


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
    def accum(self, rgb:tm.vec3):
      scaled = (rgb - self.bounds.min) / (self.bounds.max - self.bounds.min)
      gray = ti.f32(rgb_gray(scaled))
      log_gray = tm.log(tm.max(gray, 1e-4))

      ti.atomic_min(self.log_bounds.min, log_gray)
      ti.atomic_max(self.log_bounds.max, log_gray)

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
    return Metering(Bounds(vec[0], vec[1]), Bounds(vec[2], vec[3]), vec[4], vec[5], tm.vec3(vec[6], vec[7], vec[8]))


  @ti.kernel
  def metering_kernel(images: ti.types.ndarray(dtype=vec_dtype, ndim=3),
                      metering:ti.types.ndarray(dtype=vec9, ndim=0),
                      alpha:ti.f32):
    
    prev_stats = metering_from_vec(metering[None])
                      
    bounds = Bounds(np.inf, -np.inf)

    # ti.loop_config(block_dim=128)
    for i in ti.grouped(ti.ndrange(*images.shape[:3])):
      for k in ti.static(range(3)):
        bounds.expand(images[i][k])

    b = lerp(alpha, bounds.to_vec(), tm.vec2(prev_stats.bounds.min, prev_stats.bounds.max))
    bounds = Bounds(b[0], b[1])

    stats = Metering(bounds, Bounds(np.inf, -np.inf), 0, 0, tm.vec3(0, 0, 0))   

    # ti.loop_config(block_dim=128)
    for i in ti.grouped(ti.ndrange(*images.shape[:3])):
      stats.accum(images[i])

    stats.normalise(images.shape[0] * images.shape[1] * images.shape[2])
    v = lerp(alpha, stats.to_vec(), metering[None])

    metering[None] = v

  def metering_images(images, t, prev, stride=8):
      images = torch.stack(
        [image[::stride, ::stride, :] for image in images], 0)
      
      metering_kernel(images, prev, t)
      return prev
    
  @ti.kernel
  def reinhard_kernel(image: ti.types.ndarray(dtype=vec_dtype, ndim=2), 
          output : ti.types.ndarray(dtype=ti.types.vector(3, ti.u8), ndim=2),
          metering : ti.types.ndarray(dtype=ti.f32, ndim=1),
                    gamma: ti.template(),
                    intensity:ti.template(),
                    light_adapt:ti.template(),
                    color_adapt:ti.template()):
    
    stats = metering_from_vec(metering)
    log_b = stats.log_bounds
    b = stats.bounds

    max_out = 1e-6

    key = (log_b.max - stats.log_mean) / (log_b.max - log_b.min)
    map_key = 0.3 + 0.7 * tm.pow(key, 1.4)

    mean = lerp(color_adapt, stats.mean, stats.rgb_mean)

    ti.loop_config(block_dim=128)
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

      ti.atomic_max(max_out, p.max())

    ti.loop_config(block_dim=128)
    for i in ti.grouped(ti.ndrange(image.shape[0], image.shape[1])):
      p = tm.pow(image[i] / max_out, 1.0 / gamma)
      output[i] = ti.cast(255 * p, ti.u8)



  @ti.data_oriented
  class ISP():
    @beartype
    def __init__(self, bayer_pattern:bayer.BayerPattern, 
                  scale:Optional[float]=None, resize_width:int=0,
                 moving_alpha=0.1, 
                 transform:interpolate.ImageTransform=interpolate.ImageTransform.none,
                 device:torch.device = torch.device('cuda', 0),
                 metering_stride:int=8):
      
      assert scale is None or resize_width == 0, "Cannot specify both scale and resize_width"    
  
      self.bayer_pattern = bayer_pattern
      self.moving_alpha = moving_alpha
      self.scale = scale
      self.resize_width = resize_width
      self.transform = transform
      self.metering_stride = metering_stride

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
        return interpolate.resize_bilinear(image, output_size, scale)
      elif self.scale is not None:

        output_size = (round(w * self.scale), round(h * self.scale))
        return interpolate.resize_bilinear(image, output_size, self.scale)
      
      else:
        return image


    def load_16u(self, image):
      cfa = torch.empty(image.shape, dtype=torch_dtype, device=self.device)
      load_u16(image, cfa)
      return self._process_image(cfa)

    def load_16f(self, image):
      cfa = torch.empty(image.shape, dtype=torch_dtype, device=self.device)
      load_16f(image, cfa)
      return self._process_image(cfa)

    def load_packed12(self, image_data):
      w, h = (image_data.shape[1] * 2 // 3, image_data.shape[0])

      cfa = torch.empty(h, w, dtype=torch_dtype, device=self.device)    
      decode12_kernel(image_data.view(-1), cfa.view(-1))
      return self._process_image(cfa)

    def load_packed16(self, image_data):
      w, h = (image_data.shape[1] // 2, image_data.shape[0])

      cfa = torch.empty(h, w, dtype=torch_dtype, device=self.device)    
      decode16_kernel(image_data.view(-1), cfa.view(-1))
      return self._process_image(cfa)

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
      
      


    @beartype
    def tonemap_reinhard(self, images:List[torch.Tensor], 
                         gamma:float=1.0, intensity:float=1.0, light_adapt:float=1.0, color_adapt:float=0.0):

      if self.metrics is None:
        initial = torch.zeros(9, dtype=torch.float32, device=self.device)
        self.metrics = metering_images(images, 0.0, 
            initial, self.metering_stride)
      else:

        self.metrics = metering_images(images, (1.0 - self.moving_alpha), 
            self.metrics, self.metering_stride)


      outputs = [torch.empty(image.shape, dtype=torch.uint8, device=self.device) for image in images]
      for output, image in zip(outputs, images):
        reinhard_kernel(image, output, self.metrics, gamma, intensity, light_adapt, color_adapt)
      
      return transform(outputs, self.transform)

    


  ISP.__qualname__ = name
  return ISP



Camera16 = camera_isp("Camera16", ti.f16)
Camera32 = camera_isp("Camera32", ti.f32)
