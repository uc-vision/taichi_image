import taichi as ti
from taichi.math import vec2, vec3, vec4, ivec2, ivec3, ivec4
import numpy as np
import torch

u8vec3 = ti.types.vector(3, ti.u8)
u16vec3 = ti.types.vector(3, ti.u16)
f16vec3 = ti.types.vector(3, ti.f16)



scale_factor = {
  ti.u8: 255,
  ti.u16: 65535,
  ti.i16: 32767,
  ti.f16: 1.0,
  ti.f32: 1.0
}


ti_to_np = {
  ti.u8: np.uint8,
  ti.u16: np.uint16,
  ti.i16 : np.int16,  
  ti.f16: np.float16,
  ti.f32: np.float32
}

ti_to_torch = {
  ti.u8: torch.uint8,
  ti.f16: torch.float16,
  ti.f32: torch.float32
}


type_to_ti = dict(
  uint8   = ti.u8,
  uint16  = ti.u16,
  int16  = ti.i16,
  float16 = ti.f16,
  float32 = ti.f32,
)

torch_to_ti = {
  'torch.uint8'   : ti.u8,
  'torch.int16'  : ti.i16,
  'torch.float16' : ti.f16,
  'torch.float32' : ti.f32,
}

def ti_type(in_arr):
  if isinstance(in_arr, np.ndarray):
    return type_to_ti[str(in_arr.dtype)]
  elif isinstance(in_arr, torch.Tensor):
    return torch_to_ti[str(in_arr.dtype)]
  else:
    raise ValueError(f'Unsupported input type {type(in_arr)}')

def empty_array(in_arr, shape=None, dtype=None):
  shape = in_arr.shape if shape is None else shape
  dtype = ti_type(in_arr) if dtype is None else dtype


  if isinstance(in_arr, np.ndarray):
    return np.empty(shape, ti_to_np[dtype])
  elif isinstance(in_arr, torch.Tensor):
    return torch.empty(tuple(shape), dtype=ti_to_torch[dtype], device=in_arr.device)    


def zeros_array(in_arr, shape=None, dtype=None):
  shape = in_arr.shape if shape is None else shape
  dtype = ti_type(in_arr) if dtype is None else dtype
  

  if isinstance(in_arr, np.ndarray):
    return np.zeros(shape, ti_to_np[dtype])
  elif isinstance(in_arr, torch.Tensor):
    return torch.zeros(shape, ti_to_torch[dtype], device=in_arr.device)       
