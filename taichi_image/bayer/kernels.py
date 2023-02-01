import taichi as ti




@ti.kernel
def rgb_to_bayer_array(self, 
  image: ti.types.ndarray(ndim=2),
  bayer: ti.types.ndarray(ndim=2), 
                
  pixel_order: ti.template()):

  p1, p2, p3, p4 = pixel_order
  for i, j in ti.ndrange(image.shape[0] // 2, image.shape[1] // 2):
    x, y = i * 2, j * 2

    bayer[x, y] = image[x, y][p1]
    bayer[x + 1, y] = image[x + 1, y][p2]
    bayer[x, y + 1] = image[x + 1, y + 1][p3]
    bayer[x + 1, y + 1] = image[x + 1, y + 1][p4]

@ti.kernel
def bayer_to_rgb_array(bayer: ti.types.ndarray(ndim=2),
            out: ti.types.ndarray(ndim=2), in_type:ti.template(), out_type, kernels: ti.template()):

  for i, j in ti.ndrange(bayer.shape[0] // 2, bayer.shape[1] // 2):
    bayer_2x2(bayer, out, kernels, i * 2, j * 2)