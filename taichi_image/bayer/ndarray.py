
from .core import BayerPattern, bayer_kernels



def rgb_to_bayer(image, pattern:BayerPattern):
  assert image.ndim == 3 and image.shape[2] == 3, "image must be RGB"

  bayer = np.empty(image.shape[:2], dtype=np.uint8)
  rgb_to_bayer_kernel(image, bayer, pattern.pixel_order)
  return bayer




def bayer_to_rgb(bayer, pattern:BayerPattern):
  assert bayer.ndim == 2 , "image must be mono bayer"


  rgb = np.empty((*bayer.shape, 3), dtype=np.uint8)
  bayer_to_rgb_kernel(bayer, rgb, bayer_kernels(pattern))
  return rgb