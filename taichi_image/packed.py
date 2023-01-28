
import taichi as ti
import taichi.math as tm
from taichi_image.kernel import flatten, symmetrical, zip_tuple, u8vec3, u16vec3
import numpy as np


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



@ti.kernel
def encode12_kernel(values:ti.types.ndarray(ti.u16, ndim=1), 
  encoded:ti.types.ndarray(ti.u8, ndim=1)):
  for i_half in ti.ndrange(values.shape[0] // 2):
    i = i_half * 2   
    bytes = encode12_pair(u16vec2(values[i], values[i + 1]))

    idx = 3 * i_half
    for k in ti.static(range(3)):
      encoded[idx + k] = bytes[k]

    
@ti.kernel
def decode12_kernel(encoded:ti.types.ndarray(ti.u8, ndim=1), out:ti.types.ndarray(ti.u16, ndim=1)):
  for i_half in ti.ndrange(out.shape[0] // 2):
    i = i_half * 2
    idx = 3 * i_half
    bytes = ti.Vector([encoded[idx + k] for k in ti.static(range(3))], dt=ti.u8)
    pair = decode12_pair(bytes)
    out[i], out[i + 1] = pair[0], pair[1]

def encode12(values:np.ndarray):
  assert values.dtype == np.uint16

  values = values.reshape(-1)
  assert len(values) % 2 == 0, f"length must be even for 12-bit encoding got: {len(values)}"

  encoded = np.empty(((values.shape[0] * 3) // 2), dtype=np.uint8)

  encode12_kernel(values, encoded)
  return encoded


def decode12(values:np.ndarray):
  assert values.dtype == np.uint8
  assert len(values) % 3 == 0, f"length must be a factor of 3 for 12-bit decoding got: {len(values)}"

  decoded = np.empty(((values.shape[0] * 2) // 3), dtype=np.uint16)

  decode12_kernel(values, decoded)
  return decoded



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





