


from taichi_image.test.bayer import display_rgb
from test.arguments import init_with_args
import numpy as np
import cv2

from taichi_image import bayer, camera_isp
import taichi as ti

def main():
  args = init_with_args()

  test_image = cv2.imread(args.image)
  test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

  if test_image.dtype == np.uint8:
    test_image = test_image.astype(np.float32) / 255
  elif test_image.dtype == np.uint16:
    test_image = test_image.astype(np.float32) / 65535

  pattern = bayer.BayerPattern.RGGB
  test_images = [ bayer.rgb_to_bayer( (np.clip(test_image * x, 0, 1) * 65535.0).astype(np.uint16), pattern=pattern) for x in [0.2, 0.4, 0.8]]
  image_sizes = [(x.shape[1], x.shape[0]) for x in test_images]

  CameraISP = camera_isp.camera_isp(ti.f32)
  #  image_sizes:List[Tuple[int, int]], bayer_pattern:bayer.BayerPattern, resize_to:Optional[Tuple[int, int]]=None):

  isp = CameraISP(image_sizes, pattern, moving_alpha=1.0, resize_width=512)
  isp.load_16u(test_images)

  outputs = isp.outputs_like(test_images)
  isp.tonemap_reinhard(outputs, gamma=0.6, color_adapt=0.0, light_adapt=0.0)
  # isp.tonemap_linear(outputs, gamma=0.6)


  display_rgb("test", outputs[0])


if __name__ == '__main__':
  main()