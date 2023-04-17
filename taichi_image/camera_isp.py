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
      out[I] = ti.cast(image[I], dtype) / 65535.0


  vec7 = ti.types.vector(7, dtype)






  @ti.data_oriented
  class ISP():
    def __init__(self, image_sizes:List[Tuple[int, int]], 
                 bayer_pattern:bayer.BayerPattern, scale:float=1.0, 
                 moving_alpha=0.1):
      
      self.image_sizes = image_sizes
      self.num_images = len(image_sizes)
      self.bayer_to_rgb = bayer.bayer_to_rgb_func(bayer_pattern, dtype)

      self.input_cfa = [ti.field(dtype, shape=(size[1], size[0])) for size in image_sizes]
      self.input_rgb = [ti.field(rgb_dtype, shape=(size[1], size[0])) for size in image_sizes]
      
      self.scale = scale
      if scale != 1.0:

        self.output_sizes = [(round(w * scale), round(h * scale)) for w, h in image_sizes]
        self.buffer = [ti.field(rgb_dtype, shape=(size[1], size[0])) for size in self.output_sizes]
      else:
        self.output_sizes = image_sizes
        self.buffer = self.input_rgb

      self.moving_alpha = ti.field(ti.f32, shape=())
      self.moving_alpha[None] = moving_alpha

      self.running_bounds = ti.field(tonemap.Bounds, shape=(1,))
      self.running_meter = ti.field(tonemap.Metering, shape=(1,))


    def process_16u(self, images):
      for image, cfa in zip(images, self.input_cfa):
        load_u16(image, cfa)
      self.process_images_kernel()


    def process_packed12(self, images):
      for i, image in enumerate(images):
        decode12(image, self.input_cfa[i])
      self.process_images_kernel()

    
    def outputs_like(self, images):
      return [empty_like(image, shape=(size[1], size[0], 3), dtype=ti.u8) 
              for image, size in zip(images, self.output_sizes)]

    def tonemap_linear(self, outputs, gamma):
      for output, buffer in zip(outputs, self.buffer):
        self.tonemap_linear_kernel(buffer, output, gamma)

      return outputs
    
    def tonemap_reinhard(self, outputs, gamma):
      for output, buffer in zip(outputs, self.buffer):
        self.tonemap_reinhard_kernel(buffer, output, gamma)

      return outputs

    @ti.kernel
    def process_images_kernel(self):

      total_bounds = tonemap.Bounds(1e10, -1e10)
      for cfa, rgb, buffer in ti.static(zip(self.input_cfa, self.input_rgb, self.buffer)):
        self.bayer_to_rgb(cfa, rgb)

        if ti.static(self.scale != 1.0):
          interpolate.bilinear_func(rgb, buffer, self.scale, 1.0, dtype)

        total_bounds.union(tonemap.bounds_func(buffer))

      t = self.moving_alpha[None]
      self.running_bounds[None] = tonemap.bounds_from_vec(
        lerp(t, self.running_bounds[None].to_vec(), total_bounds))

      meter_total = vec7(0.0)
      for image in ti.static(self.buffer):
        m = tonemap.metering_func(image, total_bounds)
        meter_total += m.to_vec()

      self.running_meter[None] = tonemap.metering_from_vec(
        lerp(t, self.running_meter[None].to_vec(), meter_total / self.num_images))



    @ti.kernel
    def tonemap_linear_kernel(self, buffer:ti.template(), 
                       output: ti.types.ndarray(ti.types.vector(3, ti.u8), ndim=2),
                       gamma:ti.f32):
      tonemap.linear_func(buffer, output, self.running_bounds[None],
                          gamma, scale_factor=255, dtype=ti.u8)
      
    @ti.kernel
    def tonemap_reinhard_kernel(self, buffer:ti.template(), 
                         output: ti.types.ndarray(ti.types.vector(3, ti.u8), ndim=2)):
      tonemap.reinhard_func(buffer, output, self.running_bounds[None],
                            self.gamma, scale_factor=255, dtype=ti.u8)
    
          
  return ISP



