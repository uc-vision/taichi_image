


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
  test_images = [ bayer.rgb_to_bayer((test_image * x * 65536.0).astype(np.uint16), pattern=pattern) for x in [0.8, 1.0, 1.2]]
  
  image_sizes = [(x.shape[1], x.shape[0]) for x in test_images]

  CameraISP = camera_isp.camera_isp(ti.f32)
  #  image_sizes:List[Tuple[int, int]], bayer_pattern:bayer.BayerPattern, resize_to:Optional[Tuple[int, int]]=None):

  isp = CameraISP(image_sizes, pattern)
  isp.load16u(test_images)



if __name__ == '__main__':
  main()