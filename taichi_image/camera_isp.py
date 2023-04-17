from typing import List, Optional, Tuple
import taichi as ti
import taichi.math as tm
from taichi_image.types import empty_like, zeros_like

from . import tonemap, interpolate, bayer, packed
from py_structs.numpy import shape_info

import torch


def camera_isp(dtype=ti.f32):
  decode12 = packed.decode12_kernel(dtype)
  rgb_dtype = ti.types.vector(3, dtype)

  @ti.kernel
  def load_u16(image: ti.types.ndarray(ti.u16, ndim=2),
                      out: ti.template()):
    for I in ti.grouped(image):
      out[I] = ti.cast(image[I], dtype) / 65535.0
  


  @ti.data_oriented
  class ISP():
    def __init__(self, image_sizes:List[Tuple[int, int]], 
                 bayer_pattern:bayer.BayerPattern, scale:float=1.0):
      
      self.image_sizes = image_sizes
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

      self.metering = None
      self.gamma = 1.0
    

    def process_16u(self, images):
      for image, cfa in zip(images, self.input_cfa):
        load_u16(image, cfa)
      return self._process_inputs(self._outputs_like(images))

    def process_packed12(self, images):
      for i, image in enumerate(images):
        decode12(image, self.input_cfa[i])
      return self._process_inputs(self._outputs_like(images))
    
    def _outputs_like(self, images):

      return [empty_like(image, shape=(size[1], size[0], 3), dtype=ti.u8) 
              for image, size in zip(images, self.output_sizes)]

    def _process_inputs(self, outputs):
      self.process_images_kernel()

      for output, buffer in zip(outputs, self.buffer):
        self.tonemap_kernel(buffer, output)


      return outputs

    @ti.kernel
    def process_images_kernel(self):
      for cfa, rgb, buffer in ti.static(zip(self.input_cfa, self.input_rgb, self.buffer)):
        self.bayer_to_rgb(cfa, rgb)

        if ti.static(self.scale != 1.0):
          interpolate.bilinear_func(rgb, buffer, self.scale, 1.0, dtype)


    @ti.kernel
    def tonemap_kernel(self, buffer:ti.template(), 
                       output: ti.types.ndarray(ti.types.vector(3, ti.u8), ndim=2)):
      bounds = tonemap.bounds_func(buffer)
      tonemap.linear_func(buffer, output, bounds, self.gamma, scale_factor=255, dtype=ti.u8)
    
          
  return ISP



