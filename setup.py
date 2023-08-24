from setuptools import find_packages, setup

setup(name='taichi_image',
      version='0.2',
      install_requires=[
          #'taichi',
          'beartype',
          'tqdm'
      ],
      packages=find_packages(),
      entry_points={}
)
