import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Callable, List, Tuple, Optional, Union, Dict
from beartype import beartype
from natsort import natsorted
import torch
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
from taichi_image.interpolate import ImageTransform
from taichi_image.test.bayer import display_rgb, save_rgb, display_multi_rgb
from taichi_image.test.arguments import add_taichi_args

from time import perf_counter
from enum import Enum

from taichi_image import bayer, monodepth_camera_isp, camera_isp
from taichi_image.test.tonemap_utils import WeightAlgorithms, plot_histogram, plot_metrics
import taichi as ti


from monodepth_utilities.algorithms import CachedDepthAnythingV2TRTPredictor
from monodepth_utilities.utils.imports import set_depth_anything_v2_dir



def zip_fill(*args, default = None) -> list:
  """ Returns a zip of all args and fills missing items using modular indices to padd till max length. """
  sizes = [len(a) for a in args]
  count = max(sizes)
  if count == 0:
    return [] if default is None else [default]  # Return as list. 
  result = []
  for index in range(max(sizes)):
    result.append([argument[index % len(argument)] if len(argument) > 0 else default[i] for i, argument in enumerate(args)])
  return result   


class TonemapCamera(Enum):

  DEFAULT = None
  DEPTH = monodepth_camera_isp.MonoDepthCamera32
  STRIDE = camera_isp.Camera32
  

class VisualTonemap:
  """
    Visualiser for tonemap script.
  """
  def __init__(self, gamma: float, intensity: float, colour_adapt: float, light_adapt: float, threshold: float = 0.0, 
               weight: str = 'default', moving_alpha: float = 0.02, camera: str = 'default', resize_width: int = 3072,
               width: int = 4096, ids_format = None, transform: ImageTransform = ImageTransform.rotate_90, scans: List[int] = None,
               histogram: bool = False):
    self.gamma = gamma
    self.intensity = intensity
    self.colour_adapt = colour_adapt
    self.light_adapt = light_adapt
    self.width = width
    self.ids_format = ids_format
    self.camera_args = (weight, moving_alpha)
    self.threshold = threshold
    self.scans = [] if scans is None else scans
    self.histogram = histogram
    # Initialise the tone map camera.
    self.weight_func = WeightAlgorithms[weight.upper()].value()  # Initialise
    self.camera = TonemapCamera[camera.upper()]
    if self.camera == TonemapCamera.DEPTH:
      self.isp = TonemapCamera.DEPTH.value(
        bayer.BayerPattern.RGGB, transform=transform, moving_alpha=moving_alpha, resize_width=resize_width,
        stride=8, predictor=CachedDepthAnythingV2TRTPredictor(), weight_func=self.weight_func)
    elif self.camera == TonemapCamera.STRIDE:
      self.isp = TonemapCamera.STRIDE.value(
        bayer.BayerPattern.RGGB, transform=transform, moving_alpha=moving_alpha, resize_width=resize_width,
      )
    else:
      self.isp = TonemapCamera.DEFAULT.value

  @property
  def kwargs(self) -> dict:
    return dict(gamma=self.gamma, intensity=self.intensity, color_adapt=self.colour_adapt, light_adapt=self.light_adapt)
  
  @property
  def args(self) -> list:
    return self.gamma, self.intensity, self.colour_adapt, self.light_adapt, self.threshold, *self.camera_args

  def load_image(self, img_bytes: bytes) -> torch.Tensor:
    """ Load image using isp. """
    assert img_bytes.shape[0] % 2 == 0, "bytes must have an even number"
    bytes = img_bytes.view(-1, (self.width * 3) // 2)
    return self.isp.load_packed12(bytes, ids_format=self.ids_format)
  
  def execute(self, images: Dict[str, bytes], scan_id: int = -1) -> Optional[Tuple[np.ndarray, str]]:
    """ Run the tonemap algorithm on inputs. """
    if self.isp is None:
      return  # No visualiser initialised.
    if len(self.scans) > 0 and scan_id not in self.scans:
      return None  # Don't run visualiser on this particular scan.

    outputs = self.isp.tonemap_reinhard([self.load_image(img_bytes) for img_bytes in images.values()], **self.kwargs, transform=False)
    if self.threshold > 0 and self.weight_func.weights is not None:
      for output, mask in zip(outputs, self.weight_func.weights):
        if output.shape[0] != mask.shape[0] or output.shape[1] != mask.shape[1]:
          mask = F.interpolate(mask[None][None], output.shape[:2], mode='bilinear')[0, 0]
        output[mask < self.threshold, :] = 0  # Set pixels to black 
    outputs = self.isp.apply_transform(outputs)
    if self.histogram:
      outputs.insert(0, plot_histogram(self.weight_func.weights, outputs[0].shape))
    
    if True:
      outputs.insert(0, plot_metrics(self.isp.metrics, outputs[0].shape))

    return (
      np.concatenate([o.cpu().numpy() for o in outputs], axis=1),
      str(self)
    )

  def __str__(self) -> str:
    info = f'Moving Alpha: {self.camera_args[1]}, Gamma: {self.gamma}, Intensity: {self.intensity}, Colour Adapt: {self.colour_adapt}, Light Adapt: {self.light_adapt}'
    if self.camera == TonemapCamera.DEPTH:
      info = f'Weight: {str(self.weight_func)}, Threshold: {self.threshold}, ' + info
    
    return f'Mode: {self.camera.name.lower()}, ' + info
  

DEFAULT_TONEMAP = VisualTonemap(0.5, 2.0, 0.2, 1.0)  # Default parameters.


class RawScan:

  def __init__(self, folder: Union[Path, str], cam_folder: bool = False, scan_types: Optional[List[str]] = None, uid: int = -1, 
               offset: int = 0, reverse: bool = False):
    self.uid = uid
    self.directory = Path(folder)
    self.cam_folder = cam_folder
    self.scan_types = scan_types if scan_types is not None else ['.tiff', '.raw']
    self.offset = offset
    self.reverse = reverse
    self._init_raw_scan()
  
  @beartype
  def is_file_image(self, file: Path) -> bool:
    return file.is_file() and file.suffix.lower() in self.scan_types
  
  @beartype
  def find_cam_images(self, cam_id: str) -> List[str]:
    """ Find all camera images for given cam id and naturally sort them. """
    if not (camera_directory := self.directory / cam_id).is_dir():
      return []
    return natsorted([file.name for file in camera_directory.iterdir() if self.is_file_image(file)])

  def _init_raw_scan(self):
    """ Scans the raw scan directory for camera ids and common image names. """
    cam_ids = []
    common_images = None
    if self.cam_folder:
      cam_ids.append(self.directory.name)
      common_images = self.find_cam_images(cam_ids[0])
    else:
      for cam in self.directory.iterdir():
        if len(images := set(self.find_cam_images(cam.name))) > 0:
          cam_ids.append(cam.name)
          common_images = images if common_images is None else common_images & images  # & = intersection_update
    
    self.cam_ids = sorted(cam_ids)
    if common_images is None:
      raise ValueError(f"No common images found in {self.cam_ids}")

    self.cam_images = sorted(common_images)

    print(f"Found {self.cam_ids} image folders with {len(self.cam_images)} images")

  def get_cam_image(self, index: int) -> Optional[str]:
    """ Returns the camera image name at a specified index to which the offset is applied and reverse order flag. """
    real_index = len(self) - (index + self.offset + 1) if self.reverse else index + self.offset
    if 0 > real_index > len(self):
      return None
    return self.cam_images[real_index]

  def __len__(self) -> int:
    return len(self.cam_images)


@beartype
def load_raw_bytes(filepath, device:torch.device=torch.device('cuda')):
  """Load raw image bytes into torch tensor without any decoding"""
  with open(filepath, 'rb') as f:
    raw_bytes = f.read()
  return torch.frombuffer(raw_bytes, dtype=torch.uint8).to(device, non_blocking=True)


def load_images_iter(f : Callable[[Path], torch.Tensor], scans: List[RawScan]):
  """ Load a set of folders containing raw images with matching names 
       Returns an iterator of scan and image tuples {scan_id: {camera_id: raw_bytes_tensor}}
  """  
  with ThreadPoolExecutor() as executor:

    def create_load_group(index: int):
      tasks = {}
      for scan in scans:
        if image_name := scan.get_cam_image(index):
          
          tasks[scan.uid] = {cam_id: executor.submit(f, scan.directory / cam_id / image_name) for cam_id in scan.cam_ids}
      return tasks

    group = create_load_group(0)
    for i in range(1, max([len(scan) for scan in scans]) + 1):
      next_group = create_load_group(i)
      result = {scan_uid: {cam_id: future.result() for cam_id, future in scan_data.items()} 
                for scan_uid, scan_data in group.items()}
      group = next_group
      yield result


def parse_scan_params(args) -> List[RawScan]:
  """ Loads a list of raw scans based on args."""
  raw_scans = []
  if len(args.scan) > 0:
    for uid, scan_dir in enumerate(args.scan):
      if args.reverse and args.keep_order:
        raw_scans.append(RawScan(scan_dir, uid=uid, offset=args.skip_scan))
      raw_scans.append(RawScan(scan_dir, uid=uid, offset=args.skip_scan, reverse=args.reverse))
    raw_scans = [RawScan(scan_dir, uid=uid) for uid, scan_dir in enumerate(args.scan)]
  elif args.images is not None:
    if args.reverse and args.keep_order:
      raw_scans.append(RawScan(args.images, cam_folder=True, offset=args.skip_scan))
    raw_scans.append(RawScan(args.images, cam_folder=True, offset=args.skip_scan, reverse=args.reverse))
  else:
    raise ValueError("No --scan or --images specified")
  return raw_scans


def parse_tonemap_params(args) -> List[VisualTonemap]:
  """ Parse arguments into a list of tonemap parameters. """
  tonemaps = []
  camera_kwargs = dict(width=args.width, transform=args.transform, resize_width=args.resize_width, histogram=args.histogram)
  for tonemap_args in zip_fill(args.gamma, args.intensity, args.color_adapt, args.light_adapt, args.threshold, args.weight_adapt, 
                               args.moving_alpha, default=DEFAULT_TONEMAP.args):
    tonemaps.append(VisualTonemap(*tonemap_args, camera='depth', **camera_kwargs))

  if args.stride_cam:
    stride_kwargs = dict(scans=args.stride_cam_scans, camera='stride')
    params = args.stride_cam_params
    if -1 in params:  # If -1 flag is specified in params it will create stride for default params only.
      tonemaps.insert(0, VisualTonemap(*DEFAULT_TONEMAP.args, **stride_kwargs, **camera_kwargs))
    else:
      indices = [i for i in range(len(tonemaps))] if len(params) == 0 else params
      for offset, index in enumerate(indices):
        tonemaps.insert(index + offset + 1, VisualTonemap(*tonemaps[index + offset].args, **stride_kwargs, **camera_kwargs))

  return tonemaps


def print_tonemappers(visualisers: List[VisualTonemap]):
  print('Running tonemap scan with parameters: ')
  print('|   Mode   |   Gamma   |   Intensity   |   Colour Adapt   |   Light Adapt   |   Threshold   |   Weight   |   Moving Alpha   |')
  for visualiser in visualisers:
    print('|{:^10}|{:^11}|{:^15}|{:^18}|{:^17}|{:^15}|{:^12}|{:^18}|'.format(visualiser.camera.name, *visualiser.args))


def tonemap_scan_parser():
  """ Create and setup the argument parser for the tonemap scan. """
  parser = argparse.ArgumentParser()
  parser.add_argument("--scan", nargs='*', type=Path, default=[])
  parser.add_argument("--images", type=Path)

  parser.add_argument("--width", type=int, default=4096)

  # Tonemap Parameterisation.
  parser.add_argument("--gamma", nargs='*', type=float, default=[])
  parser.add_argument("--intensity", nargs='*', type=float, default=[])
  parser.add_argument("--color_adapt", nargs='*', type=float, default=[])
  parser.add_argument("--light_adapt", nargs='*', type=float, default=[])
  parser.add_argument("--threshold", nargs='*', type=float, default=[])
  parser.add_argument("--weight_adapt", nargs='*', type=str, default=[])
  parser.add_argument("--moving_alpha", nargs='*', type=float, default=[])


  parser.add_argument("--resize_width", type=int, default=3072)
  parser.add_argument("--transform", type=ImageTransform, default=ImageTransform.rotate_90)

  parser.add_argument("--stride_cam", action='store_true')  # Enables a stride camera.
  parser.add_argument("--stride_cam_params", nargs='*', type=int, default=[])  # Apply stride cam to specific parameters.
  parser.add_argument("--stride_cam_scans", nargs='*', type=int, default=[])  # Apply stride cam to specific cams.

  parser.add_argument("--keep-order", action='store_true')  # Keeps original order and reverses.
  parser.add_argument("--reverse", action='store_true')  # Reverses all scans.
  parser.add_argument("--histogram", action='store_true')  # Shows a histogram for weights.

  parser.add_argument("--dav2_module", type=Path)  # Specify the depth anything v2 module path.
  parser.add_argument("--continuous", action='store_true')  # doesn't wait for user input.
  parser.add_argument("--skip_scan", type=int, default=0)  # Skip a number of raw scan images from the start.
  add_taichi_args(parser)

  return parser


def main():
  torch.set_printoptions(precision=3, sci_mode=False, linewidth=100)
  
  args = tonemap_scan_parser().parse_args()
  
  if args.dav2_module:
    set_depth_anything_v2_dir(args.dav2_module)
    # Weights should be in dav2 folder / weights.
    CachedDepthAnythingV2TRTPredictor(weights_path=Path(args.dav2_module).resolve() / 'weights')
  
  ti.init(debug=args.debug, arch=ti.cuda if args.device == "cuda" else ti.cpu,  log_level=args.log)

  tonemappers = parse_tonemap_params(args)
  
  raw_scans = parse_scan_params(args)

  images = load_images_iter(partial(load_raw_bytes, device=torch.device(args.device)), raw_scans)

  continuous = args.continuous  # Get switchable for running viewer without input.

  try:
    # Create cameras for multiple params.
    print_tonemappers(tonemappers)
    pbar = tqdm(images, total=max([len(scan) for scan in raw_scans]))
    for index, scans_data in enumerate(pbar):
      pbar.set_description(f'Image: {index}')
      rgbs = []
      start = perf_counter()
      for scan_id, cam_images in scans_data.items():
        if len(cam_images.keys()) == 0:
          continue  # No cameras found.
        for tonemapper in tonemappers:
          if output := tonemapper.execute(cam_images, scan_id):
            rgbs.append(output)
      print(f'took to display: {perf_counter() - start} (s)')

      enter_pressed = display_multi_rgb('Tonemapped', rgbs, continuous)
      if enter_pressed:
        continuous = not continuous  # Flip flag

  except KeyboardInterrupt:
    pass


if __name__ == '__main__':
  main()