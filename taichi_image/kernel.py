import taichi as ti

def mirror(w):
  return w + w[:-1][::-1]


def symmetrical(w):
  w = mirror([mirror(row) for row in w])
  return flatten(w)

def flatten(w):
  return [x for row in w for x in row]



def kernel_square(weights, n=5):

  offsets = [ (i, j) for i in range(-(n//2), n//2 + 1) 
    for j in range(-(n//2), n//2 + 1)]

  assert len(offsets) == len(weights), f"incorrect weight length {len(offsets)} != {len(weights)}"
  return tuple(zip(offsets, weights))


def zip_tuple(*args):
  return tuple(zip(*args))



u8vec3 = ti.types.vector(3, ti.u8)
u16vec3 = ti.types.vector(3, ti.u16)
f16vec3 = ti.types.vector(3, ti.f16)


@ti.kernel
def conv(image: ti.types.ndarray(u8vec3, ndim=2), weights: ti.template(), 
  out: ti.types.ndarray(u8vec3, ndim=2)):
  total = ti.static(sum([w for _, w in weights]))

  image_size = ti.math.ivec2(image.shape)
  for i in ti.grouped(image):
    c = ti.math.vec3(0.0)
    for offset, weight in ti.static(weights):
      idx = ti.math.clamp(i + offset, 0, image_size - 1)

      c += ti.cast(image[idx], ti.f32) * weight
    out[i] = ti.cast(ti.math.clamp(c / total, 0, 255), ti.u8)
    