import matplotlib.axes
import torch
import torch.nn.functional as F
import numpy as np
import taichi as ti
import threading
import time
import queue
import cv2
import json

import matplotlib
from matplotlib.widgets import Slider, Button, CheckButtons, TextBox

from datetime import datetime
from typing import Tuple, Optional, Union, Callable, Dict, List
from enum import Enum

from pathlib import Path

from taichi_image.color import rgb_gray


# Taichi

@ti.kernel
def group_depth(depth: ti.types.ndarray(dtype=ti.f32, ndim=3), group: ti.types.ndarray(dtype=ti.int32, ndim=1)):
  """ Use taichi kernel to accumulate all depth values into a single array based on given """
  count = group.shape[0]
  for i in ti.grouped(ti.ndrange(*depth.shape)):
    index = ti.min(int(depth[i] * count), count - 1)
    group[index] += 1


@ti.kernel
def image_to_intensity_kernel(image: ti.types.ndarray(dtype=ti.types.vector(3, ti.f32), ndim=2), intensities: ti.types.ndarray(dtype=ti.f32, ndim=2)):
  """ Converts torch image to array of intensity values. """
  
  for i in ti.grouped(ti.ndrange(*image.shape[:2])):
    #intensities[i] = ti.log(rgb_gray(image[i]))
    intensities[i] = rgb_gray(image[i])

@ti.kernel
def array_to_intensity_kernel(array: ti.types.ndarray(dtype=ti.types.vector(3, ti.f32), ndim=1), intensities: ti.types.ndarray(dtype=ti.f32, ndim=1)):
  """ Compute intensity values for array of rgbs and store them in the intensity array. """
  
  for i in ti.ndrange(array.shape[0]):
    # intensities[i] = ti.log(rgb_gray(array[i]))
    intensities[i] = rgb_gray(array[i])


@ti.kernel
def update_bin_count_rgb(image: ti.types.ndarray(dtype=ti.types.vector(3, ti.u8), ndim=2), depth: ti.types.ndarray(dtype=ti.f32, ndim=2), 
                     threshold: ti.f32, bin_count: ti.types.ndarray(dtype=ti.u32, ndim=2)) -> ti.int32:
  count = 0
  bin_length = bin_count.shape[1]
  for i in ti.grouped(ti.ndrange(*image.shape[:2])):
    if depth[i] < threshold:
      continue
    count += 1
    rgb_float = image[i] / 255.0
    intensity = rgb_gray(rgb_float)
    bin_count[0, ti.min(int(rgb_float[0] * bin_length), bin_length - 1)] += 1
    bin_count[1, ti.min(int(rgb_float[1] * bin_length), bin_length - 1)] += 1
    bin_count[2, ti.min(int(rgb_float[2] * bin_length), bin_length - 1)] += 1
    bin_count[3, ti.min(int(intensity * bin_length), bin_length - 1)] += 1
  return count


def image_to_intensity(image: torch.Tensor, normalise: bool = False) -> torch.Tensor:
  """ Uses taichi kernel to compute image intensity values. """
  intensities = torch.empty(image.shape[:2], dtype=image.dtype, device=image.device)
  image_to_intensity_kernel(image, intensities)
  
  if normalise:
    intensities *= 1 / intensities.max()
  return intensities


def array_to_intensity(array: torch.Tensor, normalise: bool = False) -> torch.Tensor:
  """ Uses taichi kernel to compute array intensity values. ndim=1 of rgb values. """
  if array.dtype != torch.float32:
    array = array / 255.0
  intensities = torch.empty((array.shape[0], ), dtype=torch.float32, device=array.device)
  array_to_intensity_kernel(array, intensities)
  
  if normalise:
    intensities *= 1 / intensities.max()
  return intensities


def resize_image(image: Union[torch.Tensor, np.ndarray], new_shape: Tuple[int, int]) -> Union[torch.Tensor, np.ndarray]:
  """ Resizes image to specified shape. """
  dims = image.ndim
  shape = image.shape[1:3] if dims == 4 else image.shape[0:2]
  if shape[0] == new_shape[0] and shape[1] == new_shape[1]:
    return image  # No scaling needed.
  
  def resize(img, use_torch: bool = False):
    if use_torch:
      return F.interpolate(img, new_shape, mode='bilinear')
    return cv2.resize(img, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_LINEAR)
  
  if isinstance(image, torch.Tensor):
    if dims == 4:
      return resize(image, True)
    if dims == 3:
      return resize(image[None], True)[0]
    if dims == 2:
      return resize(image[None][None], True)[0, 0]
  return resize(image)


def threshold_image(image: Union[torch.Tensor, np.ndarray], mask: Union[torch.Tensor, np.ndarray], threshold: float) -> Union[torch.Tensor, np.ndarray]:
  """ Threshold image based on mask. """
  if isinstance(image, torch.Tensor):
    output = torch.zeros_like(image)
    mask = resize_image(mask, image.shape[0:2])
    bool_mask = mask >= threshold
    output[bool_mask] = image[bool_mask]
    return output
    
  else:
    raise NotImplementedError


class WeightAlgorithm():

  def __init__(self, *args, **kwargs):
    self.weights = None

  def __call__(self, weights: torch.Tensor) -> torch.Tensor:
    self.weights = weights
    return weights
  
  def __str__(self) -> str:
    return 'Default'
  

class WeightSquared(WeightAlgorithm):

  def __call__(self, weights: torch.Tensor) -> torch.Tensor:
    return super().__call__(weights ** 2)
  
  def __str__(self) -> str:
    return 'Squared'
  

class WeightCulled(WeightAlgorithm):

  def __init__(self, *args, threshold: float = 0.5, **kwargs):
    super().__init__(*args, **kwargs)
    self.threshold = threshold

  def __call__(self, weights):
    return super().__call__((weights >= self.threshold).to(torch.float32))



class WeightAlgorithms(Enum):

  DEFAULT = WeightAlgorithm
  SQUARE = WeightSquared


BACKGROUND_COLOUR = (243, 255, 163)


def plot_histogram(depth: Optional[torch.Tensor], shape: Tuple[int, int, int], num_bars: int = 10) -> torch.Tensor:
  """ Create a histogram using depth map onto a new torch image. """
  histogram = torch.zeros(shape, dtype=torch.uint8, device=depth.device if depth is not None else 'cuda')
  histogram[:] = torch.tensor(BACKGROUND_COLOUR, dtype=torch.uint8, device=depth.device if depth is not None else 'cuda')
  if depth is None:
    return histogram  # returns empty graph.
  group = torch.zeros((num_bars), device=depth.device, dtype=torch.int32)
  group_depth(depth, group)

  height = depth.shape[0] * depth.shape[1] * depth.shape[2]
  height_scale = histogram.shape[0] / height

  bar_width = histogram.shape[1] // num_bars
  # bar_colour = torch.tensor((0, 0, 255), dtype=torch.uint8, device=depth.device)
  for i, count in enumerate(group):
    bar_height = int(count * height_scale)
    histogram[histogram.shape[0] - bar_height:, i * bar_width: (i + 1) * bar_width] = int(i * (1.0 / num_bars) * 255)

  return histogram


def plot_metrics(metrics, shape: Tuple[int, int, int]) -> torch.Tensor:
  """ Plots metrics onto a image. """
  metric_image = np.full(shape, (243, 255, 163), dtype=np.uint8)

  metric_data = ['log_mean: ', str(metrics.log_mean), 'gray_mean: ', str(metrics.gray_mean),
                 'rgb_mean: ', str(metrics.rgb_mean[0]), str(metrics.rgb_mean[1]), str(metrics.rgb_mean[2]),
                 'map_key: ', str(metrics.map_key)]
  
  for i, data in enumerate(metric_data):
    cv2.putText(metric_image, data, (20, 600 + (i * 200)), cv2.FONT_HERSHEY_SIMPLEX, 4, (50, 50, 255), 8, 4)
  
  return torch.from_numpy(metric_image)



def single_to_rgb_channels(image: Union[torch.Tensor, np.ndarray]):
  """ Padd single channel to 3 channels for rgb image. """
  if isinstance(image, torch.Tensor):
    return torch.stack((image, ) * 3, axis=-1)
  elif isinstance(image, np.ndarray):
    return np.stack((image, ) * 3, axis=-1)
  return image


def generate_rgb_intensity_histogram(image: torch.Tensor, depth: torch.Tensor, threshold: float, bins: int = 10, normalise: bool = False) -> torch.Tensor:
  """ Generate the bin values for the histograms using taichi. """

  # channels r g b i
  bin_count = torch.zeros((4, bins), dtype=torch.uint32, device=image.device)
  count = update_bin_count_rgb(image.contiguous(), depth.contiguous(), threshold, bin_count)

  return bin_count / count if normalise else bin_count


def bins_to_min_bin(bins: List[torch.Tensor]) -> torch.Tensor:
  output = torch.zeros_like(bins[0])
  for i in range(output.shape[1]):
    for j in range(output.shape[0]):
      output[j][i] = min([bin[j][i] for bin in bins])
  return output

def bin_to_perimeter(bin: torch.Tensor, scales: List[float] = None) -> torch.Tensor:
  result = torch.zeros((bin.shape[0]), dtype=torch.float, device=bin.device)
  if scales is None:
    scales = [1.0 for _ in range(result.shape[0])]
  width = 1 / bin.shape[1]
  for i in range(bin.shape[0]):
    for j in range(bin.shape[1]):
      if bin[i][j] > 0:
        result[i] += (bin[i][j] * 2) + ((width * 2 * scales[i]))

  return result


# ---- Helper classes for matplot ----

class Metric:

  def __init__(self, axes: Optional[matplotlib.axes.Axes] = None):
    self.set_axes(axes)
    self._obj = None

  def set_axes(self, axes: Optional[matplotlib.axes.Axes]):
    self.axes = axes

  def set_value(self, value) -> None:
    pass

  def get_value(self):
    return None

  def on_changed(self, callback) -> None:
    pass

  def on_clicked(self, callback) -> None:
    pass
  
  def reset(self):
    if self._obj and hasattr(self._obj, 'reset'):
      self._obj.reset()


class MetricValue(Metric):

  def __init__(self, text: str, value = None, **kwargs):
    super().__init__(**kwargs)
    self._text = text
    self._value = value

  @property
  def text(self) -> str:
    try:
      if "{" in self._text:
        try:
          return self._text.format(*self._value)
        except TypeError:
          return self._text.format(self._value)
    except TypeError:
      pass
    return self._text + (str(self._value) if self._value is not None else '')
  
  def set_axes(self, axes):
    """ Initialise metric value. """
    super().set_axes(axes)
    if axes is None:
      self._obj = None
    else:
      axes.set_axis_off()
      self._obj = axes.text(0, 0, self.text, size=10, ha='left')

  def set_value(self, value):
    self._value = value
    if self._obj:
      self._obj.set_text(self.text)

  def get_value(self):
    return self._value
  

class MetricSlider(Metric):

  def __init__(self, title: str, value: float, min_value: float = 0.0, max_value: float = 1.0, increment: float = 0.01, **kwargs):
    super().__init__()
    self.title = title
    self.value = value
    self.min_value = min_value
    self.max_value = max_value
    self.increment = increment
    self._opts = kwargs
  
  def get_value(self):
    if self._obj:
      return self._obj.val
    return None

  def set_value(self, value):
    if self._obj:
      self._obj.set_val(value)

  def set_axes(self, axes):
    super().set_axes(axes)
    if axes is None:
      self._obj = None
    else:
      self._obj = Slider(axes, label=self.title, valmin=self.min_value, valmax=self.max_value, valinit=self.value, valstep=self.increment, **self._opts)
      # self._obj.drawon = False
  
  def on_changed(self, callback):
    if self._obj:
      self._obj.on_changed(callback)


class MetricButton(Metric):

  def __init__(self, text: str, **kwargs):
    super().__init__(**kwargs)
    self.text = text

  def set_axes(self, axes):
    super().set_axes(axes)
    if axes is None:
      self._obj = None
    else:
      self._obj = Button(axes, label=self.text)

  def on_clicked(self, callback):
    if self._obj is not None:
      self._obj.on_clicked(callback)


class MetricCheckButton(Metric):

  def __init__(self, axes=None, uid = None, **kwargs):
    super().__init__()
    self._func = None
    self._opts = kwargs
    if axes is not None:
      self.set_axes(axes)
    self.uid = uid

  def get_value(self):
    if not self._obj:
      return None
    return self._obj.get_active()
  
  def is_active(self) -> bool:
    if not self._obj:
      return False
    return len(self._obj.get_checked_labels()) > 0
  
  def set_value(self, value):
    self._obj.set_active(0, value)
  
  def set_axes(self, axes):
    super().set_axes(axes)
    if axes is None:
      self._obj = None
    else:
      self._obj = CheckButtons(axes, **{'labels': [''], 'label_props': {'fontsize': [25]}, **self._opts})
      axes.set_axis_off()
      

class MetricInput(Metric):

  def __init__(self, label: str, dtype = None, **kwargs):
    super().__init__()
    self.label = label
    self.dtype = dtype
    self._opts = kwargs

  def set_axes(self, axes):
    super().set_axes(axes)
    if axes:
      self._obj = TextBox(axes, self.label, **self._opts)
      self._value = self._obj.text
      self._obj.on_text_change(self.validate)
    else:
      self._obj = None
  
  def get_value(self):
    if not self._obj:
      return None
    val = self._obj.text.get_text()
    if self.dtype:
      return self.dtype(val)
    return val
  
  def validate(self, v):
    if self.dtype is None:
      return
    try:
      parsed_value = self.dtype(v)
      self._value = parsed_value
    except ValueError:
      parsed_value = self._value
    self._obj.set_val(str(parsed_value))
    

class Visual:

  def __init__(self, title: str = '', title_args = None, **kwargs):
    self._obj = None
    self._title = title
    self._title_args = None
    self._title_obj = None
    self._axes = None

  @property
  def title(self) -> str:
    if len(self._title) == 0:
      return ''
    if self._title_args:
      return self._title.format(*[str(arg) for arg in self._title_args])
    return self._title

  def set_title_args(self, *args) -> None:
    self._title_args = args
    if self._title_obj:
      self._title_obj.set_text(self.title)

  def set_axes(self, axes: matplotlib.axes.Axes) -> None:
    self._axes = axes
    self._title_obj = axes.set_title(self.title)  # Update title.

  def set_value(self, data) -> None:
    pass


class VisualImage(Visual):

  def set_value(self, data: Union[torch.Tensor, np.ndarray]):
    if isinstance(data, torch.Tensor):
      data = data.cpu().numpy()

    if self._obj:
      self._obj.set_data(data)
    elif self._axes:
      self._obj = self._axes.imshow(data)


class VisualHisto(Visual):

  def __init__(self, title = '', title_args=None, bins: int = 40, bin_range: Optional[tuple] = None, data_is_binned: bool = False, **kwargs):
    super().__init__(title, title_args, **kwargs)
    self._options = {}
    self._max = None
    self._bins = bins
    self._bin_range = bin_range
    self._data_bin_flag = data_is_binned

  def set_max(self, value):
    self._max = value
    if self._max is not None and self._axes is not None:
      self._axes.set_ylim(0, self._max)

  def get_max(self):
    return self._max

  def _get_bins_step(self) -> float:
    if self._bin_range is None:
      return 1.0 / self._bins
    return (self._bin_range[1] - self._bin_range[0]) / self._bins

  def _generate_bins(self, start: float, step: float, size: int):
    return np.array([start + (i * step) for i in range(size + 1)])

  def _get_bins(self):
    if self._bin_range is None:
      return self._bins
    return self._generate_bins(self._bin_range[0], self._get_bins_step(), self._bins)

  def set_options(self, **kwargs):
    if 'bin_range' in kwargs:
      self._bin_range = kwargs.pop('bin_range')
    if 'bins' in kwargs:
      self._bins = kwargs.pop('bins')
    self._options = {**self._options, **kwargs}

  def set_value(self, data: Union[torch.Tensor, np.ndarray]):
    """ Set histogram data. """
    if not self._axes:
      return

    if isinstance(data, torch.Tensor):
      data = data.cpu().numpy()
    
    if data.ndim > 1:
      data = data.flatten()

    self._axes.cla()
    opts = {'bins': self._get_bins()}
    normalise = self._options.pop('normalise', False)
    xlim = self._options.pop('xlim', None)
    if self._data_bin_flag:
      bins = opts['bins']
      if isinstance(bins, int):
        bins = self._generate_bins(0, 1 / bins, bins)
      self._axes.hist(bins[:-1], len(bins) - 1, weights=data)
    else:
      if normalise and len(data) > 0:
        opts['weights'] = np.full(len(data), 1 / len(data))
      self._axes.hist(data, **{**self._options, **opts})
    self._axes.set_title(self.title)  # Redrat title too since cleaerd.
    if xlim is not None:
      self._axes.set_xlim(xlim)
    self.set_max(self._max)


class VisualOverlapHisto(VisualHisto):

  def __init__(self, title='', title_args=None, **kwargs):
    super().__init__(title, title_args, **kwargs)
    self._options = []
    self.overlap = 0.0
    self.overlap_area = 0.0
    self.overlap_perimeter = 0.0

  def set_options(self, options: List[dict]):
    bin_ranges = []
    bin_sizes = []
    for opt in options:
      if 'bin_range' in opt:
        bin_ranges.append(opt.pop('bin_range'))
      if 'bins' in opt:
        bin_sizes.append(opt.pop('bins'))
    if len(bin_ranges) > 0:
      if not all([bin_ranges[0] == bin_range for bin_range in bin_ranges]):
        raise AttributeError('Unable to set bin ranges as multiple are provided that do not match.')
      self._bin_range = bin_ranges[0]
    if len(bin_sizes) > 0:
      if not all([bin_sizes[0] == bin_size for bin_size in bin_sizes]):
        raise AttributeError('Unable to set bins as multiple are provided that do not match.') 
      self._bins = bin_sizes[0]
    self._options = options

  def _update_overlap(self, data: List[np.ndarray]):
    """ Update the overlaps. """
    overlap_area = 0
    overlap_total = 0
    overlap_perimeter = 0
    # width = self._get_bins_step()
    width = 1 / self._bins  # Use normalised.
    for i in range(self._bins):
      height = min(data[j][i] for j in range(len(data)))
      overlap_total += height
      overlap_area += (height * width)  # Area
      overlap_perimeter += ((2 * height) + (2 * width))
    self.overlap = overlap_total
    self.overlap_area = overlap_area
    self.overlap_perimeter = overlap_perimeter

  def set_value(self, data: Union[list[Union[torch.Tensor, np.ndarray]], torch.Tensor]):
    if not self._axes:
      return
    
    self._axes.cla()
    data_hist = []
    for i, hdata in enumerate(data):
      if isinstance(hdata, torch.Tensor):
        hdata = hdata.cpu().numpy()

      if hdata.ndim > 1:
        hdata = hdata.flatten()

      opts = self._options[i] if i < len(self._options) else {}
      opts['bins'] = self._get_bins()
      normalise = opts.pop('normalise', False)
      if self._data_bin_flag:
        bins = opts['bins']
        if isinstance(bins, int):
          bins = self._generate_bins(0, 1 / bins, bins)
        self._axes.hist(bins[:-1], len(bins) - 1, weights=hdata)
        data_hist.append(hdata)
      else:
        if normalise and len(hdata) > 0:
          opts['weights'] = np.full(len(hdata), 1 / len(hdata))
        data_hist.append(self._axes.hist(hdata, **opts)[0])
    self._axes.set_title(self.title)
    if not normalise:
      self.set_max(max([d.max() for d in data_hist]))

    self._update_overlap(data_hist)
    self._axes.legend(loc='upper right', labels=[f'Overlap: {self.overlap * 100} %', f'Overlap Perimeter {self.overlap_perimeter}'])


class VisualLinesGraph(Visual):

  def __init__(self, x_min: float = 0.0, x_max: float = 1.0, y_min: float = 0.0, y_max: float = 0.0, lines: Dict[str, dict] = None, 
               x_label: str = '', y_label: str = '', **kwargs):
    super().__init__(**kwargs)
    self.x_min = x_min
    self.x_max = x_max
    self.y_min = y_min
    self.y_max = y_max
    self.lines = lines
    self.x_label = x_label
    self.y_label = y_label

    self._lines_data = {line_name: {} for line_name in self.lines.keys()}

  def update(self, y_value, x_values: dict[str, float]):
    """ Update lines data with given values."""
    for x_key, x_value in x_values.items():
      if (line_data := self._lines_data.get(x_key, None)) is not None:
        line_data[y_value] = x_value
    
    # Plot line.
    self._axes.cla()
    self._axes.set_ylim(self.y_min, self.y_max)
    self._axes.set_xlim(self.x_min, self.x_max)

    self._axes.set_xlabel(self.x_label)
    self._axes.set_ylabel(self.y_label)

    for line, line_data in self._lines_data.items():
      x_data, y_data = [], []
      for y, x in line_data.items():
        y_data.append(y)
        x_data.append(x)
      self._axes.plot(y_data, x_data, **self.lines.get(line, {}))

  def get_y_max_index(self, line: str) -> int:
    data = self._lines_data[line]
    data_max, data_index = -1, -1
    for x, y in data.items():
      if y > data_max:
        data_max = y
        data_index = x
    return data_index

# thread

def delayed_executor(delay: float = 1.0) -> Callable:
  """ Creates a function that can execute a delayed function. """
  fps = 60
  frame_time = delay / fps
  
  def delay(f, q: queue.Queue):
    i = 0
    while i < fps:
      time.sleep(frame_time)
      try:
        q.get_nowait()
        i = 0  # Reset timer.
        while q.get_nowait():
          pass
      except queue.Empty:
        i += 1
        pass
    # f()

  thread = None
  delay_queue = queue.Queue()

  func = None

  def finish():
    nonlocal func
    nonlocal thread
    print('wait for htread.')
    thread.join()
    print('done waiting for thread.')
    thread = None
    if func is not None:
      func()  # Execute final function.

  def execute(f):
    nonlocal thread
    nonlocal func
    func = f
    if thread is None:
      thread = threading.Thread(target=delay, args=(finish, delay_queue))
      thread.run()
      print('creating new thread...')
    else:
      delay_queue.put(True)
  
  return execute



class DelayedExecutor:

  def __init__(self, delay):
    self.timer = None
    self.delay = delay

  def execute(self, func):
    if self.timer is not None:
      print('cancel call')
      self.timer.cancel()

    self.timer = threading.Timer(self.delay, func)
    self.timer.start()


class ResultLogger:

  def __init__(self, output_folder: Optional[Path] = None, save_every: int = 100):
    self.output_folder = output_folder
    self.save_every = save_every
    self._folder_name = None
    self._buffer_count = 0
    self._buffer = []
    self._save_count = 0
  
  def save(self):
    if self._folder_name is None:
      return
    folder = Path(self.output_folder) if self.output_folder is not None else Path()
    output_folder = folder / self._folder_name
    output_folder.mkdir(parents=True, exist_ok=True)
    with open(f'results-{str(self._save_count).zfill(5)}.json', 'r') as f:
      json.dump(self._buffer, f)
    self._save_count += 1
    self._buffer = []
    self._buffer_count = 0

  def initialise(self):
    self.save()
    self._save_count = 0
    self._folder_name = datetime.now().strftime('%d-%m-%y-%H-%M-%S')
  
  def log(self, data):
    print('logging data...', self._buffer_count, self.save_every)
    self._buffer.append(data)
    self._buffer_count += 1
    if self._buffer_count == self.save_every:
      self.save()


