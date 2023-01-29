import taichi as ti

def kernels(in_type, out_type):

  in_vec3 = ti.types.vector(3, in_type)
  out_vec3 = ti.types.vector(3, out_type)


  @ti.data_oriented
  class Kernels:


    @ti.kernel
    def rgb_to_bayer_kernel(self, image: ti.types.ndarray(in_vec3, ndim=2),
                    bayer: ti.types.ndarray(ti.u8,
                                            ndim=2), pixel_order: ti.template()):

      p1, p2, p3, p4 = pixel_order
      for i, j in ti.ndrange(image.shape[0] // 2, image.shape[1] // 2):
        x, y = i * 2, j * 2

        bayer[x, y] = image[x, y][p1]
        bayer[x + 1, y] = image[x + 1, y][p2]
        bayer[x, y + 1] = image[x + 1, y + 1][p3]
        bayer[x + 1, y + 1] = image[x + 1, y + 1][p4]


  return Kernels()