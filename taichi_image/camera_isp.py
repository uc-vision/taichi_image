from typing import List, Optional, Tuple
import taichi as ti
import taichi.math as tm

from . import tonemap, interpolate, bayer, packed
from abc import ABCMeta, abstractmethod

import torch


def camera_isp(dtype=ti.f32):
  decode12 = packed.decode12_kernel(dtype)

  @ti.kernel
  def load_u16(image: ti.types.ndarray(ti.u16, ndim=2),
                      out: ti.template()):
    for I in ti.grouped(image):
      out[I] = ti.cast(image[I], dtype) / 65535.0
  


  @ti.data_oriented
  class ISP():
    def __init__(self, image_sizes:List[Tuple[int, int]], bayer_pattern:bayer.BayerPattern, resize_to:Optional[Tuple[int, int]]=None):
      
      self.image_sizes = image_sizes
      self.resize_to = resize_to

      self.bayer_to_rgb = bayer.bayer_to_rgb_func(bayer_pattern, dtype)

      self.input_cfa = [ti.field(dtype, shape=(size[1], size[0])) for size in image_sizes]
      self.input_rgb = [ti.field(ti.types.vector(3, dtype), shape=(size[1], size[0])) for size in image_sizes]
      
      if resize_to is not None:
        self.buffer = [ti.field(dtype, shape=(resize_to[1], resize_to[0], 3)) for _ in image_sizes]
      else:
        self.buffer = self.input_rgb

      self.metering = None
      self.gamma = 1.0
    

    def load16u(self, images):
      for image, cfa in zip(images, self.input_cfa):
        load_u16(image, cfa)

      self.process()

    def load12(self, images):
      for i, image in enumerate(images):
        decode12(image, self.input_cfa[i])

      self.process()

    @ti.kernel
    def process(self):
      for cfa, rgb, buffer in ti.static(zip(self.input_cfa, self.input_rgb, self.buffer)):
        self.bayer_to_rgb(cfa, rgb)

        if ti.static(self.resize_to != None):
          interpolate.bilinear_func(rgb, buffer, dtype)

        

    @ti.kernel
    def tonemap(self, buffer:ti.template(), output: ti.types.ndarray(ti.types.vector(3, ti.u8), ndim=2)):
      bounds = tonemap.bounds_func(buffer)
      tonemap.linear_func(buffer, output, bounds, self.gamma)
    

    def outputs(self):
      outputs = [torch.empty((size[1], size[0], 3), dtype=torch.uint8) for size in self.image_sizes]

      for outputs, buffer in zip(outputs, self.buffer):
        self.tonemap(buffer, outputs)
      
          
  return ISP



