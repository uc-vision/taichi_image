from typing import List, Optional, Tuple
import taichi as ti
import taichi.math as tm
from taichi_image.types import empty_like
from torch import lerp

from . import tonemap, interpolate, bayer, packed


def camera_isp(dtype=ti.f32):
  decode12 = packed.decode12_kernel(dtype)
  rgb_dtype = ti.types.vector(3, dtype)

  @ti.kernel
  def load_u16(image: ti.types.ndarray(ti.u16, ndim=2),
                      out: ti.template()):
    for I in ti.grouped(image):
      x = ti.cast(image[I], ti.f32) / 65535.0
      out[I] = ti.cast(x, dtype) 

  @ti.kernel
  def load_16f(image: ti.types.ndarray(ti.u16, ndim=2),
                      out: ti.template()):
    for I in ti.grouped(image):
      out[I] = ti.cast(image[I], dtype)



  vec7 = ti.types.vector(7, ti.f32)



  @ti.data_oriented
  class ISP():
    def __init__(self, num_images:int, image_size:Tuple[int, int], bayer_pattern:bayer.BayerPattern, 
                  scale:Optional[float]=None, resize_width:int=0,
                 moving_alpha=0.1):
      
      self.image_size = image_size
      self.num_images = num_images
      
      self.bayer_to_rgb = bayer.bayer_to_rgb_func(bayer_pattern, dtype)

      self.input_cfa = [ti.field(dtype, shape=(image_size[1], image_size[0])) for _ in range(num_images)]
      self.input_rgb = [ti.field(rgb_dtype, shape=(image_size[1], image_size[0])) for _ in range(num_images)]

      assert scale is None or resize_width == 0, "Cannot specify both scale and resize_width"

      if scale is not None:

        self.scale = scale
        self.output_size = (round(w * scale), round(h * scale))
        self.buffer = [ti.field(rgb_dtype, shape=(output_size[1], output_size[0])) for _ in range(num_images)]

      elif resize_width > 0:

        self.scale = resize_width / image_size[0] 
        self.output_size = (resize_width, round(h * self.scale))

        self.buffer = [ti.field(rgb_dtype, shape=(output_size[1], output_size[0])) for _ in range(num_images)]

      else:

        self.output_size = image_size
        self.buffer = self.input_rgb
        self.scale = None


      self.moving_alpha = ti.field(ti.f32, shape=())
      self.moving_alpha[None] = moving_alpha

      self.running_weight = ti.field(ti.f32, shape=())
      self.running_weight[None] = 0.0

      self.running_bounds = tonemap.Bounds.field(shape=())
      self.running_bounds[None] = tonemap.Bounds(0, 1)

      self.running_meter = tonemap.Metering.field(shape=())


    def load_16u(self, images):
      for image, cfa in zip(images, self.input_cfa):
        load_u16(image, cfa)
      self._process_images_kernel()


    def load_16f(self, images):
      for image, cfa in zip(images, self.input_cfa):
        load_16f(image, cfa)
      self._process_images_kernel()


    def load_packed12(self, images):
      for i, image in enumerate(images):
        decode12(image, self.input_cfa[i])
      self._process_images_kernel()

    
    def outputs_like(self, images):
      return [empty_like(image, shape=(self.output_size[1], self.output_size[0], 3), dtype=ti.u8) 
              for image in images]


    @ti.kernel
    def _process_images_kernel(self):

      total_bounds = tonemap.Bounds(1e10, -1e10)
      for cfa, rgb, buffer in ti.static(zip(self.input_cfa, self.input_rgb, self.buffer)):
        self.bayer_to_rgb(cfa, rgb)

        if ti.static(self.scale != None):
          interpolate.bilinear_func(rgb, buffer, self.scale, 1.0, dtype)

        total_bounds.union(tonemap.bounds_func(buffer))

      alpha = self.moving_alpha[None]
      t = ti.min(alpha / (self.running_weight[None] + alpha), alpha)
      self.running_weight[None] += alpha


      self.running_bounds[None] = tonemap.bounds_from_vec(
        tonemap.lerp(t, self.running_bounds[None].to_vec(), total_bounds.to_vec()))
      
      
      meter_total = vec7(0.0)
      for image in ti.static(self.buffer):
        tonemap.rescale_func(image, self.running_bounds[None], dtype)

        m = tonemap.metering_func(image)
        meter_total += m.to_vec()

      self.running_meter[None] = tonemap.metering_from_vec(
        tonemap.lerp(t, self.running_meter[None].to_vec(), meter_total / self.num_images))



    def tonemap_linear(self, outputs, gamma=1.0):
      for output, buffer in zip(outputs, self.buffer):
        self._tonemap_linear_kernel(buffer, output, gamma)

      return outputs
    
    def tonemap_reinhard(self, outputs, gamma=1.0, intensity=1.0, light_adapt=1.0, color_adapt=0.0):
      for output, buffer in zip(outputs, self.buffer):
        self._tonemap_reinhard_kernel(buffer, output, gamma, intensity, light_adapt, color_adapt)

      return outputs

    @ti.kernel
    def _tonemap_linear_kernel(self, buffer:ti.template(), 
                       output: ti.types.ndarray(ti.types.vector(3, ti.u8), ndim=2),
                       gamma:ti.f32):
      tonemap.gamma_func(buffer, output, gamma, scale_factor=255, dtype=ti.u8)
      
    @ti.kernel
    def _tonemap_reinhard_kernel(self, buffer:ti.template(), 
                         output: ti.types.ndarray(ti.types.vector(3, ti.u8), ndim=2),
                         gamma:ti.f32, intensity:ti.f32, light_adapt:ti.f32, color_adapt:ti.f32):
      
      tonemap.reinhard_func(buffer, self.running_meter[None], intensity, light_adapt, color_adapt, dtype)

      bounds = tonemap.bounds_func(buffer)
      tonemap.linear_func(buffer, output, bounds, gamma, scale_factor=255, dtype=ti.u8)
    
          
  return ISP



