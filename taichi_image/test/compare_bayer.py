import colour_demosaicing 
import argparse
from taichi_image.bayer import BayerPattern, bayer_to_rgb, rgb_to_bayer

from taichi_image.test.arguments import init_with_args
from taichi_image.test.bayer import display_rgb
import numpy as np
import cv2


def trim_image(image):
  if image.shape[0] % 2 == 1:
    image = image[:-1]
  
  if image.shape[1] % 2 == 1:
    image = image[:, :-1]
  
  return image


def main():
    args = init_with_args()
    
    test_image = cv2.imread(args.image, cv2.IMREAD_COLOR)  
    test_image = trim_image(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))


    pattern = BayerPattern.RGGB
    converted_rgb = rgb_to_bayer(test_image, pattern)

    algorithms = dict(
        taichi_image = lambda image: bayer_to_rgb(image, pattern),
        opencv = lambda image: cv2.cvtColor(image, cv2.COLOR_BayerBG2RGB),
        bilinear = lambda image: colour_demosaicing.demosaicing_CFA_Bayer_bilinear(image, pattern.name),
        malvar = lambda image: colour_demosaicing.demosaicing_CFA_Bayer_Malvar2004(image, pattern.name),
        menon = lambda image: colour_demosaicing.demosaicing_CFA_Bayer_Menon2007(image, pattern.name),
    )

    for k, algorithm in algorithms.items():
        result = algorithm(converted_rgb)
        result = np.clip(result, 0, 255).astype(test_image.dtype)

        psnr = cv2.PSNR(test_image, result)
        display_rgb(f"{k} - {psnr:.3f}", result)

if __name__ == "__main__":
    main()