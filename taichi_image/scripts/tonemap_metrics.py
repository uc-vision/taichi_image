import argparse
import itertools
import matplotlib.pyplot as plt
import numpy as np
import taichi as ti
import torch

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, Dict, Tuple, List

from taichi_image import bayer, camera_isp
from taichi_image.monodepth_camera_isp import MonoDepthCamera32
from taichi_image.scripts.tonemap_scan import DEFAULT_TONEMAP, load_raw_bytes
from taichi_image.interpolate import ImageTransform
from taichi_image.test.tonemap_utils import (image_to_intensity, array_to_intensity, single_to_rgb_channels, threshold_image, 
                                             resize_image, generate_rgb_intensity_histogram, MetricCheckButton, MetricValue, MetricSlider, VisualOverlapHisto,
                                              MetricButton, MetricInput, VisualHisto, VisualImage, WeightAlgorithm, WeightCulled,
                                              VisualLinesGraph, bin_to_perimeter, bins_to_min_bin, ResultLogger)


from monodepth_utilities.algorithms import CachedDepthAnythingV2TRTPredictor
from monodepth_utilities.utils.imports import set_depth_anything_v2_dir


@dataclass
class MetricParameterisation:

  uid: str
  value: float
  min: float = 0.0
  max: float = 1.0


CAMERA_KWARGS = dict(bayer_pattern=bayer.BayerPattern.RGGB, transform=ImageTransform.rotate_90, resize_width=3072)


def load_images(images: List[Path], isp, width: int = 4096) -> Dict[str, torch.Tensor]:
  """ Load image into a torch tensor. """
  with ThreadPoolExecutor() as executor:
    futures = {image.as_posix(): executor.submit(load_raw_bytes, image) for image in images}
    image_bytes = {name: future.result() for name, future in futures.items()}
  loaded_images = {}
  for name, img_bytes in image_bytes.items():
    assert img_bytes.shape[0] % 2 == 0, "bytes must have an even number"
    packed_bytes = img_bytes.view(-1, (width * 3) // 2)
    loaded_images[name] = (isp.load_packed12(packed_bytes))
  return loaded_images


def setup_metrics_figure(run_tonemap = None) -> Callable:
  """
    Setup widgets for tonemap.
  """
  fig = plt.figure()
  fig.text(.1, .95, 'Metrics', size=14)

  metrics = dict(
    intensity_threshold = MetricSlider('intensity threshold', 0.0, 0.0, 1.0),
    depth_threshold = MetricSlider('depth threshold', 0.5, 0.0, 1.0),
    gamma = MetricSlider('gamma', DEFAULT_TONEMAP.gamma, 1e-4, 3.0),
    light_adapt = MetricSlider('light adapt', DEFAULT_TONEMAP.light_adapt, 1e-4, 3.0),
    colour_adapt = MetricSlider('colour adapt', DEFAULT_TONEMAP.colour_adapt, 1e-4, 3.0),
    intensity = MetricSlider('intensity', DEFAULT_TONEMAP.intensity, 1e-4, 3.0),
    # bounds_min = MetricSlider('bounds min', 1e-4, 1e-4, 1),
    bounds_max = MetricSlider('bounds max', 1.0, 0.0, 1.0)
  )
  
  def get_metric_data() -> dict:
    return {name: metric.get_value() for name, metric in metrics.items() if metric.get_value() is not None}

  reset_flag = False  # Flag to prevent multiple update calls at once when reset.

  def on_changed():
    nonlocal reset_flag
    if run_tonemap and not reset_flag:
      run_tonemap(metrics=get_metric_data())

  def reset_func():
    nonlocal reset_flag
    reset_flag = True
    for metric in metrics.values():
      metric.reset()
    reset_flag = False
      
  height = .6 / (len(metrics) + 1)
  
  params = []
  for i, (uid, metric) in enumerate(metrics.items()):
    y = .95 - ((i + 1) * height)
    axes = fig.add_axes([0.25, y, .5, height])
    metric.set_axes(axes)

    ax = fig.add_axes([.85, y, .1, height])
    params.append(MetricCheckButton(ax, uid=uid))

    # metric.on_changed(on_changed)
  y = .95 - ((len(metrics) + 1) * height)
  reset_axes = fig.add_axes([0.1, y, .35, height])
  metrics['reset'] = MetricButton('Reset')
  metrics['reset'].set_axes(reset_axes)
  metrics['reset'].on_clicked(lambda v: reset_func())

  update_axes = fig.add_axes([0.55, y, 0.35, height])
  metrics['update'] = MetricButton('Update')
  metrics['update'].set_axes(update_axes)
  metrics['update'].on_clicked(lambda v: on_changed())

  # Paramerisation

  fig.text(.1, .29, 'Parameterisation', size=14, verticalalignment='center')

  localise_axes = fig.add_axes([.1, .2, .4, .06])
  localise_input = MetricCheckButton(localise_axes, labels=['Localise params.'], label_props={'fontsize': [14]})

  visualise_axes = fig.add_axes([.5, .2, .4, .06])
  visualise_input = MetricCheckButton(visualise_axes, labels=['Show Histograms'], label_props={'fontsize': [14]})

  iter_axes = fig.add_axes([.4, .10, .3, .1])
  iter_input = MetricSlider('Iterations', 20, 1, 100, 1)
  iter_input.set_axes(iter_axes)

  progress_axes = fig.add_axes([.1, .02, .8, .075])
  progress_bar = MetricSlider('', 0, 0, 100, dragging=False, initcolor=None, color='green', handle_style={'size': 0})
  progress_bar.set_axes(progress_axes)

  progress_text = fig.text(.5, .05 + (.075 / 2), '', ha='center', va='center')


  def start(val):
    nonlocal localise_input, visualise_input, params, metrics
    data = get_metric_data()
    active_metrics = {p.uid: MetricParameterisation(p.uid, metrics[p.uid].get_value(), metrics[p.uid].min_value, metrics[p.uid].max_value) for p in params if p.is_active()}

    run_tonemap(metrics=data, parameterisation={
      'metrics': active_metrics, 
      'state': 'start', 
      'localise': localise_input.is_active(), 
      'visualise': visualise_input.is_active(),
      'iter_count': iter_input.get_value()
    })

  def stop(val): 
    run_tonemap(parameterisation={'state': 'stop'})

  run_axes = fig.add_axes([0.1, .1, .15, .1])
  run_button = MetricButton('Run')
  run_button.set_axes(run_axes)
  run_button.on_clicked(start)

  stop_axes = fig.add_axes([.75, .1, .15, .1])
  stop_button = MetricButton('Stop')
  stop_button.set_axes(stop_axes)
  stop_button.on_clicked(stop)


  # Outside update.

  def update(params: dict):
    # prevent items from being garbage collected.
    nonlocal run_button, stop_button, progress_text, progress_bar, iter_input

    for key, value in params.items():
      if key in metrics:
        metrics[key].set_value(value)
    
    if (iteration := params.get('iteration')) is not None:
      max_iteration = params.get('max_iteration', 1)
      progress_bar.set_value(iteration / max_iteration * 100)
      progress_text.set_text(f'{iteration} / {max_iteration}')

    fig.canvas.draw_idle()

  return update


def setup_tonemap_figure(image: str) -> Callable:
  """ Setup images and histogram visuals for the tonemapping. """
  fig = plt.figure()
  fig.suptitle(f'Tonemapping: {image}')

  visuals = dict (
    intensity_histo = VisualHisto('Intensity Histogram'),
    intensity_foreground_histo = VisualHisto('Intensity Foreground Histogram'),
    intensity_background_histo = VisualHisto('Intensity Background Histogram'),
    output = VisualImage('Output'),
    scan = VisualImage('Scan Image'),
    depth = VisualImage('Depth Map'),
    mask = VisualImage('Depth Culled Scan (threshold: {})'),
    intensity = VisualImage('Intensity Image.'),
  )

  metrics = dict(
    log_mean = MetricValue('log mean: {:.5}'),
    rgb_mean = MetricValue('rgb mean: ({:.5}, {:.5}, {:.5})'),
    gray_mean = MetricValue('gray mean: {:.5}'),
    map_key = MetricValue('map key: {:.5}'),
  )

  visuals['intensity_histo'].set_axes(fig.add_axes([0.05, .7, .15, .25]))
  visuals['intensity_foreground_histo'].set_axes(fig.add_axes([0.05, .375, .15, .25])) 
  visuals['intensity_background_histo'].set_axes(fig.add_axes([0.05, 0.05, .15, .25]))
  visuals['output'].set_axes(fig.add_axes([.25, .05, .3, .75]))
  visuals['scan'].set_axes(fig.add_axes([.6, .55, .15, .4]))
  visuals['depth'].set_axes(fig.add_axes([.8, .55, .15, .4]))
  visuals['mask'].set_axes(fig.add_axes([.6, 0.05, .15, .4]))
  visuals['intensity'].set_axes(fig.add_axes([.8, 0.05, .15, .4]))

  metrics['log_mean'].set_axes(fig.add_axes([.25, .835, .125, .04]))
  metrics['gray_mean'].set_axes(fig.add_axes([.25, .91, .125, .04]))
  metrics['rgb_mean'].set_axes(fig.add_axes([.425, .835, .125, .04]))
  metrics['map_key'].set_axes(fig.add_axes([.425, .91, .125, .04]))

  def update(result: dict):
    """ Update the visuals. """
    metrics['log_mean'].set_value(result['log_mean'])
    metrics['gray_mean'].set_value(result['gray_mean'])
    metrics['map_key'].set_value(result['map_key'])
    metrics['rgb_mean'].set_value(result['rgb_mean'])
    
    visuals['scan'].set_value(torch.rot90(result['image'], k=3))
    intensities = torch.rot90(image_to_intensity(result['image']), k=3)
    visuals['intensity'].set_value(single_to_rgb_channels(((intensities - intensities.min()) / (intensities.max() - intensities.min()))))
    visuals['intensity_histo'].set_value(intensities)
    if (weights := result.get('weights')) is not None:
      visuals['depth'].set_value(single_to_rgb_channels(weights))
      visuals['mask'].set_value(threshold_image(result['output'], weights, result['threshold']))
      visuals['intensity_foreground_histo'].set_value(intensities[weights >= result['threshold']])
      visuals['intensity_background_histo'].set_value(intensities[weights < result['threshold']])
    else:
      depth_image = torch.rot90(single_to_rgb_channels(intensities), k=3)
      visuals['depth'].set_value(depth_image)
      visuals['mask'].set_value(depth_image)
      visuals['intensity_foreground_histo'].set_value(intensities)
      visuals['intensity_background_histo'].set_value(intensities)
    
    fig.canvas.draw_idle()
  
  return update


def setup_tonemap_images() -> Callable:
  """ Setup a figure for all conjoined images for easy viewing. """
  fig = plt.figure()
  fig.suptitle('Tonemap outputs')

  visuals = []

  def update(params: dict):
    outputs = [result['output'] for result in params['tonemap_results'].values()]
    if len(visuals) == 0:
      width = 1.0 / len(outputs)
      for i in range(len(outputs)):
        visual = VisualImage()
        visual.set_axes(fig.add_axes([i * width, 0, width, 1.0]))
        visuals.append(visual)
    
    for visual, output in zip(visuals, outputs):
      visual.set_value(output)

    fig.canvas.draw_idle()
  return update


def setup_tonemap_histos() -> Callable:
  """ Setup a figure for al contjoined images to display their r g b histo grams and intensity of the thresholded image. """
  fig = plt.figure()
  plt.suptitle('Tonemap Histograms')

  histos = []

  def update(params: dict):
    outputs = [r['output'] for r in params['tonemap_results'].values()]

    if len(histos) == 0:
      width = (1.0 - ((len(outputs) + 2) * .05)) / (len(outputs) + 1)
      for x in range(len(outputs) + 1):
        height = .75 / 4
        output_histos = []
        for y in range(4):
          histo = VisualHisto(data_is_binned=True) if x < len(outputs) else VisualOverlapHisto(bin_range=(0, 255) if y > 0 else (0, 1.0), data_is_binned=True)
          histo.set_axes(fig.add_axes([(.05 * x) + .05 + (x * width), (.05 * y) + .05 + (y * height), width, height]))
          output_histos.append(histo)

        histos.append(output_histos)

    # This is also slow.
    reds, greens, blues, intensities = [], [], [], []
    for output_histos, binned_data in zip(histos, params['binned_outputs']):
      output_histos[0].set_options(color='.7', normalise=True, xlim=(0, 1.0), bin_range=(0, 1.0))
      output_histos[1].set_options(color='red', normalise=True, xlim=(0, 260), bin_range=(0, 255))
      output_histos[2].set_options(color='green', normalise=True, xlim=(0, 260), bin_range=(0, 255))
      output_histos[3].set_options(color='blue', normalise=True, xlim=(0, 260), bin_range=(0, 255))

      output_histos[0].set_value(binned_data[3])
      output_histos[1].set_value(binned_data[0])
      output_histos[2].set_value(binned_data[1])
      output_histos[3].set_value(binned_data[2])

      intensities.append(binned_data[3])
      reds.append(binned_data[0])
      blues.append(binned_data[1])
      greens.append(binned_data[2])

    inc = .7 / len(outputs)
    histos[-1][0].set_options([{'normalise': True, 'bins': 40, 'color': str((i + 1) * inc), 'alpha': 0.5} for i in range(len(outputs))])
    histos[-1][1].set_options([{'normalise': True, 'bins': 40, 'color': (.2 + (i + 1) * inc, .2, .2, 0.5)} for i in range(len(outputs))])
    histos[-1][2].set_options([{'normalise': True, 'bins': 40, 'color': (.2, .2 + (i + 1) * inc, .2, 0.5)} for i in range(len(outputs))])
    histos[-1][3].set_options([{'normalise': True, 'bins': 40, 'color': (.2, .2, .2 + (i + 1) * inc, 0.5)} for i in range(len(outputs))])
    
    histos[-1][0].set_value(intensities)
    histos[-1][1].set_value(reds)
    histos[-1][2].set_value(greens)
    histos[-1][3].set_value(blues)

    # update max values of all single axes.
    for output_histos in histos[:-1]:
      for i in range(len(output_histos)):
        output_histos[i].set_max(histos[-1][i].get_max())

    fig.canvas.draw_idle()
  
  return update


def run_parameterisation(metrics: dict, parameterisation: dict, run_tonemap: Callable):
  """ Run the parameterisation. """
  active_metrics = parameterisation['metrics'].values()
  max_iteration = 1
  steps = []
  for metric in active_metrics:
    if parameterisation.get('localise', False):
      metric_range = (metric.max - metric.min) * .2
      metric_start = max(metric.value - metric_range, metric.min)
      metric_end = min(metric.value + metric_range, metric.max)
    else:
      metric_start = metric.min
      metric_end = metric.max
    count = parameterisation.get('iter_count', 10)
    max_iteration *= count
    step = (metric_end - metric_start) / count
    steps.append( [metric_start + (i * step) for i in range(count + 1)])


  for i, permutation in enumerate(itertools.product(*steps)):
    params = {
      'metrics': metrics,
      'visualise': parameterisation.get('visualise', False),
      'iteration': i,
      'max_iteration': max_iteration,
      'param_name': ", ".join([m.uid for m in active_metrics])
    }
    # Update the metrics before passed to tonemap.
    for m, v in zip(active_metrics, permutation):
      params['metrics'][m.uid] = v
    results = run_tonemap(params)

    yield results


def evaluate_tonemap() -> Callable:
  """ Put data into bins. """

  result_logger = ResultLogger(save_every=25)

  def update(param):
    param_iteration = param.get('iteration')
    if param_iteration is not None and param_iteration <= 1:
      result_logger.initialise()
    overlaps = {'value': 0.0}
    bins = []
    for result in param['tonemap_results'].values():
      bins.append(generate_rgb_intensity_histogram(result['output'], result['weights'], result['threshold'], 40, True))

    bin = bins_to_min_bin(bins)
    bin_perimeter = bin_to_perimeter(bin).cpu().numpy()
    for overlap, data in zip(['red', 'green', 'blue', 'intensity'], bin_perimeter):
      overlaps[overlap] = data
      overlaps['value'] += data

    param['tonemap_score'] = overlaps
    param['binned_outputs'] = bins
    
    if param_iteration is not None:
      result_logger.log({'metrics': param['metrics'], 'tonemap_score': {k: float(v) for k, v in overlaps.items()}})
    
      if param_iteration == param['max_iteration']:
        result_logger.save()

    return param
  
  return update


def setup_metric_parameterisation() -> Callable:
  fig = plt.figure()
  fig.suptitle('Parameterisation')

  line_options = {
    'y_label': 'Overlap value',
    'y_max': 20,
    'lines': {
      'red': {'color': 'red'},
      'green': {'color': 'green'},
      'blue': {'color': 'blue'},
      'intensity': {'color': '0.5'},
      'value': {'color': 'purple'}
    }
  }

  line_axes = fig.add_axes([0.1, 0.1, .8, .8])
  line_visual = None

  def update(param: dict):
    nonlocal line_visual

    if not 'iteration' in param:
      return

    if param['iteration'] <= 1:
      line_axes.cla()
      line_visual = VisualLinesGraph(**line_options, x_max=param['max_iteration'], x_label=param['param_name'])
      line_visual.set_axes(line_axes)

    if not line_visual:
      return
    
    # overlaps['value'] /= 4
    line_visual.update(param['iteration'], param['tonemap_score'])

    if param['visualise'] or param['iteration'] == param['max_iteration']:
      fig.canvas.draw_idle()

  return update


def setup_tonemap(image_paths: List[Path], depth_anything_v2_module: Optional[Path] = None) -> Callable:
  """ 
    Setup the tonamp data to run. 
    Returns a function that can be called to update the current metrics provided new values.
  """
  # Setup camera
  ti.init(debug=False, arch=ti.cuda)

  monodepth = depth_anything_v2_module is not None
  weight_model = WeightCulled()
  if not monodepth:
    isp = camera_isp.Camera32(**CAMERA_KWARGS)
  else:
    set_depth_anything_v2_dir(depth_anything_v2_module)
    predictor = CachedDepthAnythingV2TRTPredictor(weights_path=Path(depth_anything_v2_module).resolve() / 'weights')
    isp = MonoDepthCamera32(**CAMERA_KWARGS, predictor=predictor, weight_func=weight_model)
    
  # load image to run tonemapping metrics on.
  images = load_images(image_paths, isp)

  eval_tonemap = evaluate_tonemap()

  # Create callback function.
  def run_tonemap(params: dict) -> dict:
    metrics = params['metrics']

    if hasattr(isp.weight_func, 'threshold'):
      isp.weight_func.threshold = metrics['depth_threshold']
    params['tonemap_results'] = {}
    for name, image in images.items():
      # Clone the original image so we don't override and get different results each iteration.
      tonemap_image = image.clone()
      if monodepth:
        isp.reset()
        # isp.metrics.bounds_min = params['bounds_min']
        isp.metrics.bounds_max = metrics['bounds_max']
      else:
        isp.metrics = None
      
      output = isp.tonemap_reinhard([tonemap_image], metrics['gamma'], metrics['intensity'], metrics['light_adapt'], metrics['colour_adapt'])[0]

      if monodepth:
        weights = torch.rot90(resize_image(weight_model.weights[0, :, :], tonemap_image.shape[0:2]), k=3)
        tonemap_metrics = {'log_mean': isp.metrics.log_mean, 'rgb_mean': isp.metrics.rgb_mean, 
                   'gray_mean': isp.metrics.gray_mean, 'map_key': isp.metrics.map_key}
      else:
        weights = None
        tonemap_metrics = {'log_mean': -1, 'rgb_mean': (-1, -1, -1), 'gray_mean': -1, 'map_key': -1}

      params['tonemap_results'][name] = {
        **tonemap_metrics,
        'threshold': metrics['depth_threshold'],
        'output': output,
        'weights': weights,
        'image': tonemap_image
      }

    return eval_tonemap(params)

  return run_tonemap


def tonemap_metrics_parser():
  """ Create argument parser for metrics. """
  parser = argparse.ArgumentParser()

  # parser.add_argument("--image", type=Path, default='/mnt/maara/raw/scan_10-32-17/cam1/00121.raw')
  inference = False
  drive = '/mnt/maara' if inference else '/uc/research/CropVision'
  scans = [
    '/raw/scan_10-32-17/cam1/00121.raw', '/raw/scan_10-52-18/cam1/00416.raw', '/raw/scan_14-05-05/cam1/00121.raw',
    '/raw/exposures/scan_14-12-11/cam1/00090.raw', '/raw/exposures/scan_14-13-49/cam1/00090.raw', '/raw/exposures/scan_14-16-03/cam1/00090.raw'
  ]
  parser.add_argument("--image", action='extend', nargs='+', type=Path, default=[Path(drive + scan) for scan in scans])
  parser.add_argument("--dav2_module", type=Path, default='/local/kla129/Depth-Anything-V2')
  parser.add_argument("--parameterisation", action='store_true')
  parser.add_argument("--headless", action='store_true')

  return parser


def main():

  args = tonemap_metrics_parser().parse_args()
  # Weights should be in dav2 folder / weights.
  headless = args.headless
  run_tonemap = setup_tonemap(args.image, Path(args.dav2_module))

  if headless:
    update_callbacks = {}
  else:
    update_callbacks = {
      'images': setup_tonemap_images(),
      'histos': setup_tonemap_histos(),
      'parameterisation': setup_metric_parameterisation(),
      'figures': {
        name.as_posix(): setup_tonemap_figure(name.as_posix()) for name in args.image
      }
    }

  # Update visuals and run the tonemapping section for metrics.

  parameterisation_running  = False

  def update_matplot(params):
    """ Update matplot lib visuals. """
    # Updates the individual tonemap metric plots.
    if (update_metrics := update_callbacks.get('metrics')) is not None:
      update_metrics(params)

    if params.get('visualise', True):
      for name, metric_data in params['tonemap_results'].items():
        update_tonemaps = update_callbacks.get('figures', {})
        if update_tonemap_metric := update_tonemaps.get(name):
          update_tonemap_metric(metric_data)
      update_callbacks['histos'](params)
      update_callbacks['images'](params)
    update_callbacks['parameterisation'](params)



  def tonemap(metrics: Optional[dict] = None, parameterisation: Optional[dict] = None):
    """ """
    nonlocal parameterisation_running
    if parameterisation is None:
      if metrics is None:
        return
      return update_matplot(run_tonemap({'metrics': metrics}))
    if parameterisation.get('state') == 'start':
      parameterisation_running = True
      for params in run_parameterisation(metrics, parameterisation, run_tonemap):
        if not headless:
          update_matplot(params)
          plt.pause(.01)
        if parameterisation_running == False:
          break
    elif parameterisation.get('state') == 'stop':
      parameterisation_running = False

  if not headless:
    update_callbacks['metrics'] = setup_metrics_figure(tonemap)

    plt.show()
  else:
    # Run without matplotlib.
    pass


if __name__ == '__main__':

  main()
