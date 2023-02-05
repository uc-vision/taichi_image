import taichi as ti
from taichi_image.packed import encode12, decode12

import numpy as np

def test_encode_decode(n=1000):
  for i in range(100):

    size = np.random.randint(n)*2

    x = np.random.randint(0, 2**12, size=size, dtype=np.uint16)
    encoded = encode12(x)
    decoded = decode12(encoded)

    assert np.all(x == decoded)


def main():
  ti.init(arch=ti.cuda, log_level=ti.DEBUG)

  test_encode_decode()


if __name__ == '__main__':
  main()