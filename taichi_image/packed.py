
import taichi as ti
import taichi.math as tm
from taichi_image.kernel import flatten, symmetrical, zip_tuple, u8vec3, u16vec3
import numpy as np
from taichi_image import types
from taichi_image.util import cache


u16vec2 = ti.types.vector(2, ti.u16)

@ti.func
def encode12_pair(pixels: u16vec2) -> u8vec3:
  # 2 x 12 bits -> 3 x 8 bits 

  return ti.Vector([
    (pixels[0] >> 4) & 0xff, 
    (pixels[0] & 0xf) << 4 | (pixels[1] >> 8), 
    pixels[1] & 0xff,
  ], dt=ti.u8)


@ti.func
def decode12_pair(input:u8vec3) -> u16vec2:
  # 3 x 8 bits -> 2 x 12 bits (as u16)

  bytes = ti.cast(input, ti.u16)
  return ti.Vector([
    ((bytes[0] & 0xff) << 4) | (bytes[1] >> 4),
    ((bytes[1] & 0xf) << 8) | (bytes[2] & 0xff),
    ], dt=ti.u16)

@cache
def encode12_kernel(in_type, scaled=False):
  scale = types.scale_factor[in_type]
  

  @ti.func
  def read_value_scaled(arr:ti.template(), i:ti.int32) -> ti.u16:
    value = ti.cast(arr[i], ti.f32) * (4095.0 / scale)
    rounded =  ti.round(value, ti.u16)
    return rounded
  
  @ti.func
  def read_value_direct(arr:ti.template(), i:ti.int32) -> ti.u16:
    return arr[i]
  
  read_value = read_value_scaled if scaled else read_value_direct


  @ti.kernel
  def f(values:ti.types.ndarray(in_type, ndim=1), 
    encoded:ti.types.ndarray(ti.u8, ndim=1)):
    for i_half in ti.ndrange(values.shape[0] // 2):
      i = i_half * 2   
      bytes = encode12_pair(u16vec2(read_value(values, i), read_value(values, i + 1)))

      idx = 3 * i_half
      for k in ti.static(range(3)):
        encoded[idx + k] = bytes[k]

  return f

@cache 
def decode12_func(out_type, scaled=False):
  scale = types.scale_factor[out_type]
  

  @ti.func
  def write_value_scaled(arr:ti.template(), i:ti.int32, value:ti.u16):
    val_float = ti.cast(value, ti.f32) * (scale / 4095.0)
    arr[i] = ti.cast(val_float, out_type)

  @ti.func
  def write_value_direct(arr:ti.template(), i:ti.int32, value:ti.u16):
    arr[i] = value

  write_value = write_value_scaled if scaled else write_value_direct

  @ti.func
  def decode(encoded:ti.types.ndarray(ti.u8, ndim=1), out:ti.types.ndarray(out_type, ndim=1)):
    for i_half in ti.ndrange(out.shape[0] // 2):
      i = i_half * 2
      idx = 3 * i_half
      bytes = ti.Vector([encoded[idx + k] for k in ti.static(range(3))], dt=ti.u8)
      pair = decode12_pair(bytes)

      write_value(out, i, pair[0])
      write_value(out, i + 1, pair[1])

  return decode

@cache
def decode12_kernel(out_type, scaled=False):
  f = decode12_func(out_type, scaled=scaled)

  @ti.kernel
  def k(encoded:ti.types.ndarray(ti.u8, ndim=1), 
        out:ti.types.ndarray(out_type, ndim=1)):
    f(encoded, out)
      
  return k



def encode12(values, scaled=False):
  shape = values.shape
  assert shape[-1] % 2 == 0, f"last dimension must be even for 12-bit encoding got: {shape}"
  values = values.reshape(-1)

  encoded = types.empty_array(values, ((values.shape[0] * 3) // 2), dtype=ti.uint8)
  f = encode12_kernel(types.ti_type(values), scaled=scaled)
    
  f(values, encoded)
  return encoded.reshape(shape[:-1] + (shape[-1] * 3 // 2,))


def decode12(values, dtype=ti.u16, scaled=False):
  shape = values.shape
  assert types.ti_type(values) == ti.uint8
  assert shape[-1] % 3 == 0, f"last dimension must be a factor of 3 for 12-bit decoding got: {shape}"
  values = values.reshape(-1)

  decoded = types.empty_array(values, ((values.shape[0] * 2) // 3), dtype=dtype)
  f = decode12_kernel(dtype, scaled=scaled)

  f(values, decoded)
  return decoded.reshape(shape[:-1] + (shape[-1] * 2 // 3,))



@ti.data_oriented
class PackedMono12:
  def __init__(self, shape):
    assert shape[1].value % 2 == 0, "height must be even for 12-bit encoding"
    
    self.shape = shape
    self.packed = ti.field(dtype=ti.u8, shape=(shape[0], shape[1] * 3 // 2))


  @ti.func
  def __getitem__(self, p:tm.ivec2) -> tm.vec3:
    i = (p.x // 2) * 3
    bytes = u8vec3(self.packed[p.y, i], self.packed[p.y, i + 1], self.packed[p.y, i + 2])
    p1, p2 = decode12_pair(bytes)
    return ti.select(p.x % 2 == 0, p1, p2)





