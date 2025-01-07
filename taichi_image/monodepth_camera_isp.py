from typing import List, Optional, Union, Tuple
from beartype import beartype
import taichi as ti
import taichi.math as tm
from taichi_image.color import rgb_gray
from taichi_image.types import ti_to_torch
import torch
import torch.nn.functional as F

from taichi_image.util import Bounds, lerp, vec9, vec6, vec5

from . import tonemap, interpolate, bayer, packed
from .camera_isp import moving_average


from monodepth_utilities.algorithms import Predictor
from monodepth_utilities.utils.tranforms import resize_image


# constants

DepthIndex = Optional[Union[int, slice]]
BatchSize = Optional[int]

GRAY_LOG_MIN: int = 1e-4   # Bottom bounds for converting rgb to logged gray scale


def camera_isp(name:str, dtype=ti.f32):
  decode16_kernel = packed.decode16_kernel(dtype, scaled=True)

  torch_dtype = ti_to_torch[dtype]
  vec_dtype = ti.types.vector(3, dtype)


  @ti.kernel
  def load_u16(image: ti.types.ndarray(ti.u16, ndim=2),
               out:   ti.types.ndarray(dtype, ndim=2)):
    for i in ti.grouped(image):
      x = ti.cast(image[i], ti.f32) / 65535.0
      out[i] = ti.cast(x, dtype) 

  @ti.kernel
  def load_16f(image: ti.types.ndarray(ti.u16, ndim=2),
              out:   ti.types.ndarray(dtype, ndim=2)):
    for i in ti.grouped(image):
      out[i] = ti.cast(image[i], dtype)


  @ti.data_oriented
  class Metrics:
    """
      Metric data oriented class for interaction between taichi and input metrics.
    """

    @ti.dataclass
    class Metering:
      """
        Metrics Metering struct for computing metrics based on image and weight array.
      """
      log_mean: ti.f32
      mean: ti.f32
      rgb_mean: tm.vec3
      total_weight: ti.f32
      total_count: ti.f32

      @ti.func
      def to_vec(self):
        return vec5(self.log_mean, self.mean, *self.rgb_mean)
      
      @ti.func 
      def accum(self, rgb: tm.vec3, weight: ti.f32):
        gray = ti.f32(rgb_gray(rgb))
        log_gray = tm.log(tm.max(gray, GRAY_LOG_MIN))

        self.log_mean += (log_gray * weight)
        self.mean += (gray * weight)
        self.rgb_mean += rgb
        self.total_weight += weight
        self.total_count += 1.0

      @ti.func
      def normalise(self):
        self.log_mean /= self.total_weight
        self.mean /= self.total_weight
        # rgb values not affacted by the weight.
        self.rgb_mean /= self.total_count


    def __init__(self, device):
      """ Initialise the metrics class. """
      self._values = torch.zeros(5, device=device, dtype=torch.float32)
      self._weights = torch.zeros((1, 1, 1), device=device, dtype=torch.float)
      self.log_bounds_max = 1.0
      self.log_bounds_min = -9.210340371976182  # math.log(1e-4)

    @property
    def values(self):
      return self._values
    
    @property
    def log_mean(self):
      """ Return the log mean average based on current metrics. """
      return self._values[0].item()
    
    @property
    def gray_mean(self):
      """ Return the gray mean based on current metrics. """
      return self._values[1].item()
    
    @property
    def rgb_mean(self):
      """ Returns the rgb mean as a vector 3 tensor based on current metrics. """
      return (self._values[2].item(), self._values[3].item(), self._values[4].item())
    
    @property
    def map_key(self):
      """ Return the map key based on current metrics. """
      key = (self.log_bounds_max - self.log_mean) / (self.log_bounds_max - self.log_bounds_min)
      return 0.3 + (0.7 * (key ** 1.4))
    
    def reset(self) -> None:
      """ Reset metrics. """
      #self._data[None] = vec5(0, 0, 0, 0, 0)
    
    def update(self, inputs: torch.Tensor, alpha: float, weights: Optional[torch.Tensor] = None):
      """ Update the moving metering. """
      weights_flag = True
      if weights is None:
        weights = self._weights
        weights_flag = False
      self.kernel(inputs, weights, weights_flag, alpha, self._values)
    
    @ti.kernel
    def kernel(self, 
               inputs: ti.types.ndarray(dtype=vec_dtype, ndim=3), 
               weights: ti.types.ndarray(dtype=ti.f32, ndim=3), 
               weights_flag: ti.u1, 
               alpha: ti.f32, 
               metering: ti.types.ndarray(dtype=vec5, ndim=0)):
      """ 
        Taichi kernel for accumulating the pixel values and normalising based on weights if flag is set. 
        Updates the provided metering vector.
      """
      stats = Metrics.Metering(0, 0, tm.vec3(0, 0, 0), 0, 0)
      for i in ti.grouped(ti.ndrange(*inputs.shape[:3])):
        stats.accum(inputs[i], weights[i] if weights_flag == 1 else 1.0)
      stats.normalise()
      v = lerp(alpha, stats.to_vec(), metering[None])
      metering[None] = v


  def initialise_inputs(images: List[torch.Tensor], size: int, idx: DepthIndex) -> torch.Tensor:
    """ Formats the images into torch tensor for the model to run predictions on. """
    inputs = torch.stack(images)
    if idx is not None:
      inputs = inputs[idx]
      if len(inputs.shape) == 3:
        inputs = inputs[None]  # Add dimension
    inputs = inputs.permute(0, 3, 1, 2)  # N H W C  ->  N C H W
    if inputs.dtype == torch.uint8:
      # Convert to float before interpolate.
      inputs = inputs / 255.0
    return resize_image(inputs, size, size, 14)  # depth anything v2 uses 14 stride.


  def inputs_to_depth(inputs: torch.Tensor, batch_size: BatchSize, predictor: Predictor) -> Tuple[torch.Tensor, torch.Tensor]:
    """ 
      Run the model prediction on the torch inputs denoted as N C H W and 
      return the inputs as N H W C and normalised results as floats 0 -> 1 
      A specified batch size can be given to predict the depth in batches.
    """
    if batch_size is None:
      results = predictor.predict(inputs)
    else:
      assert inputs.shape[0] % batch_size == 0, 'Unable to batch inputs into batch size. len(inputs) ' + \
        f'({inputs.shape[0]}) is not divisible by batch size ({batch_size}).'
      results = torch.concat(
        [predictor.predict(split_input) for split_input in inputs.split(batch_size)]
      )
    normalised_depth = (results - results.min()) / (results.max() - results.min())
    if len(normalised_depth.shape) == 4:
      if normalised_depth.shape[1] == 1:
        # N C H W -> N H W
        normalised_depth = normalised_depth[:, 0]
      elif normalised_depth.shape[3] == 1:
        # N H W C -> N H W
        normalised_depth = normalised_depth[:, :, :, 0]
    return inputs.permute(0, 2, 3, 1), normalised_depth  # Normalised result

    
  @ti.kernel
  def reinhard_kernel(image: ti.types.ndarray(dtype=vec_dtype, ndim=2), 
                      output : ti.types.ndarray(dtype=ti.types.vector(3, ti.u8), ndim=2),
                      map_key: ti.f32,
                      gray_mean: ti.f32,
                      rgb_mean: ti.types.vector(3, dtype=ti.float32),
                      gamma: ti.template(),
                      intensity:ti.template(),
                      light_adapt:ti.template(),
                      color_adapt:ti.template()):
    """
      Run the reinhard kernel on the input image and store the results in the provided output array.
    """
    max_out = 1e-6
    mean = lerp(color_adapt, gray_mean, rgb_mean)

    ti.loop_config(block_dim=128)
    for i in ti.grouped(ti.ndrange(image.shape[0], image.shape[1])):
      rgb = image[i]
      gray = rgb_gray(image[i])
      
      # Blend between gray value and RGB value
      adapt_color = lerp(color_adapt, tm.vec3(gray), rgb)

      # Blend between mean and local adaptation
      adapt_mean = lerp(light_adapt, mean, adapt_color)
      adapt = tm.pow(tm.exp(-intensity) * adapt_mean, map_key)

      p = rgb * (1.0 / (adapt + rgb))
      image[i] = ti.cast(p, dtype)

      ti.atomic_max(max_out, p.max())

    ti.loop_config(block_dim=128)
    for i in ti.grouped(ti.ndrange(image.shape[0], image.shape[1])):
      p = tm.pow(image[i] / max_out, 1.0 / gamma)
      output[i] = ti.cast(255 * p, ti.u8)

  @ti.kernel
  def linear_kernel(image: ti.types.ndarray(dtype=vec_dtype, ndim=2), 
          output : ti.types.ndarray(dtype=ti.types.vector(3, ti.u8), ndim=2),
          metering : ti.types.ndarray(dtype=ti.f32, ndim=1),
                    gamma: ti.template()):
    
    # stats = metering_from_vec(metering)
    stats = None
    tonemap.linear_func(image, output, stats.bounds, gamma, 255, ti.u8)
    

  class ISP():
    @beartype
    def __init__(self, 
                 bayer_pattern: bayer.BayerPattern, 
                 scale: Optional[float] = None, 
                 resize_width: int = 0,
                 moving_alpha: float = 0.1, 
                 transform: interpolate.ImageTransform = interpolate.ImageTransform.none,
                 device: torch.device = torch.device('cuda', 0),
                 size: int = 518,
                 idx: DepthIndex = None,  # Number of images to use for depth metering. -1 is all and any positive value limits.
                 batch_size: BatchSize = None,  # Number of images to use when prediction is run. None is all
                 stride: int = 0,  # If stride is > 0 initial metering will use this.
                 debug: bool = False,   # debug flag.
                 predictor: Optional[Predictor] = None,  # Monodepth predictor to use
                 ):
      
      assert scale is None or resize_width == 0, "Cannot specify both scale and resize_width"    
  
      self.bayer_pattern = bayer_pattern
      self.moving_alpha = moving_alpha
      self.scale = scale
      self.resize_width = resize_width
      self.transform = transform
      self.size = size
      self.idx = idx
      self.stride = stride
      self.batch_size = batch_size

      self.device = device
      self.metrics = Metrics(self.device)
      self.initalised = False

      self.predictor = predictor
      if self.predictor is None:
        raise RuntimeError('Unable to run depth camera without valid predictor.')

      self.debug = debug
      self.masks = []
      self.visuals = []

    def has_masks(self) -> bool:
      return len(self.masks) > 0
    
    def has_visuals(self) -> bool:
      return len(self.visuals) > 0
    
    def get_transformed_masks(self) -> list[torch.Tensor]:
      if not self.has_masks():
        return []
      return [interpolate.transform(mask, self.transform) for mask in self.masks]
    
    def get_transformed_visuals(self) -> list[torch.Tensor]:
      if not self.has_visuals():
        return []
      return [interpolate.transform(visual, self.transform) for visual in self.visuals]

    def reset(self) -> None:
      """ Reset the initialisation which will reset the metrics. """
      self.metrics.reset()
      self.masks = []
      self.visuals = []
      self.initalised = False

    @beartype
    def set(self, moving_alpha:Optional[float]=None, resize_width:Optional[int]=None, 
              scale:Optional[float]=None, 
              transform:Optional[interpolate.ImageTransform]=None):
      if moving_alpha is not None:
        self.moving_alpha = moving_alpha

      if resize_width is not None:
        self.resize_width = resize_width
        self.scale = None

      if scale is not None:
        self.scale = scale
        self.resize_width = 0

      if transform is not None:
        self.transform = transform
      

    def resize_image(self, image):
      w, h = image.shape[1], image.shape[0]
      if self.resize_width > 0:

        scale = self.resize_width / w 
        output_size = (self.resize_width, round(h * scale))
        return interpolate.resize_bilinear(image, output_size, scale)
      elif self.scale is not None:

        output_size = (round(w * self.scale), round(h * self.scale))
        return interpolate.resize_bilinear(image, output_size, self.scale)
      
      else:
        return image


    def load_16u(self, image):
      cfa = torch.empty(image.shape, dtype=torch_dtype, device=self.device)
      load_u16(image, cfa)
      return self._process_image(cfa)

    def load_16f(self, image):
      cfa = torch.empty(image.shape, dtype=torch_dtype, device=self.device)
      load_16f(image, cfa)
      return self._process_image(cfa)

    def load_packed12(self, image_data, ids_format=False):

      decode12_kernel = packed.decode12_kernel(dtype, scaled=True, ids_format=ids_format)
      w, h = (image_data.shape[1] * 2 // 3, image_data.shape[0])

      cfa = torch.empty(h, w, dtype=torch_dtype, device=self.device)    
      decode12_kernel(image_data.view(-1), cfa.view(-1))
      return self._process_image(cfa)

    def load_packed16(self, image_data):
      w, h = (image_data.shape[1] // 2, image_data.shape[0])

      cfa = torch.empty(h, w, dtype=torch_dtype, device=self.device)    
      decode16_kernel(image_data.view(-1), cfa.view(-1))
      return self._process_image(cfa)

    def updated_bounds(self, bounds:List[Bounds]):
      bounds = tonemap.union_bounds(bounds)
      self.moving_bounds = moving_average(self.moving_bounds, tonemap.bounds_to_np(bounds), self.moving_alpha)
      return self.moving_bounds
    
    def updated_metrics(self, image_metrics:List[vec9]):    
      mean_metrics = sum(image_metrics) / len(image_metrics)
      self.moving_metrics = moving_average(self.moving_metrics, mean_metrics, self.moving_alpha)
      return self.moving_metrics
        
    def _process_image(self, cfa):
      rgb = bayer.bayer_to_rgb(cfa)
      return self.resize_image(rgb) 
    
    def update_metrics_stride(self, images: List[torch.Tensor], alpha: float) -> None:
      """ Update metrics based on image stride. """
      images = torch.stack([image[::self.stride, ::self.stride, :] for image in images], 0)
      self.metrics.update(images, alpha)
      # metering_kernel_weights(images, metering, alpha, self._weights))

    def update_metrics_depth(self, images: List[torch.Tensor], alpha: float, idx: DepthIndex) -> None:
      """ Update the metrics based on depth model. """
      #  def metering_images_depth(images, t, prev, size = 518, idx: DepthIndex = None, batch_size: BatchSize = None):
      inputs = initialise_inputs(images, self.size, idx)
      inputs, weights = inputs_to_depth(inputs, self.batch_size, self.predictor)
      self.metrics.update(inputs, alpha, weights)
      
    #@time_cuda_func
    def update_metering(self, images:List[torch.Tensor], initial: bool = False):
      """ 
        Update the metrics from the provided images.
        If the metering function is initial and the class has not been initialised then it will run
        the initial metrics with alpha 0 and stride if a stride is provided.
        Otherwise the metrics are computed using a depth model.
      """
      if initial:
        if self.initalised:
          return
        if self.stride > 0:
          self.update_metrics_stride(images, 0.0)
        else:
          self.update_metrics_depth(images, 0.0, len(images) // 2)  # Use center image index for initial
        self.initalised = True
      else:
        self.update_metrics_depth(images, 1.0 - self.moving_alpha, self.idx)

    @beartype
    def tonemap_reinhard(self, images:List[torch.Tensor], 
                         gamma:float=1.0, intensity:float=1.0, light_adapt:float=1.0, color_adapt:float=0.0):
      self.update_metering(images, True)  # Initial metering when warming up.
      outputs = [torch.empty(image.shape, dtype=torch.uint8, device=self.device) for image in images]

      for output, image in zip(outputs, images):
        reinhard_kernel(image, output, self.metrics.map_key, self.metrics.gray_mean, self.metrics.rgb_mean, gamma, intensity, light_adapt, color_adapt)

      self.update_metering(outputs)  # Metering based on tonemap results.
      return [interpolate.transform(output, self.transform) for output in outputs]

    @beartype
    def tonemap_linear(self, images:List[torch.Tensor],  gamma:float=1.0):
      # self.update_metering(images)
      self.update_metering(images, True)
      outputs = [torch.empty(image.shape, dtype=torch.uint8, device=self.device) for image in images]
      for output, image in zip(outputs, images):
        linear_kernel(image, output, self.metrics, gamma)
      self.update_metering(outputs)
      
      return [interpolate.transform(output, self.transform) for output in outputs]


  ISP.__qualname__ = name
  return ISP



MonoDepthCamera16 = camera_isp("Camera16", ti.f16)
MonoDepthCamera32 = camera_isp("Camera32", ti.f32)
