import os
import collections


from ._common_blocks import ChannelSE
from .. import get_submodules_from_kwargs
from ..weights import load_model_weights

backend = None
layers = None
models = None
keras_utils = None

ModelParams = collections.namedtuple(
    'ModelParams',
    ['model_name', 'repetitions', 'residual_block', 'attention']
)


# -------------------------------------------------------------------------
#   Helpers functions
# -------------------------------------------------------------------------

def handle_block_names(stage, block):
  name_base = 'stage{}_unit{}_'.format(stage + 1, block + 1)
  conv_name = name_base + 'conv'
  bn_name = name_base + 'bn'
  relu_name = name_base + 'relu'
  sc_name = name_base + 'sc'
  return conv_name, bn_name, relu_name, sc_name


def get_conv_params(**params):
  default_conv_params = {
      'kernel_initializer': 'he_uniform',
      'use_bias': False,
      'padding': 'valid',
  }
  default_conv_params.update(params)
  return default_conv_params


def get_bn_params(**params):
  axis = 3 if backend.image_data_format() == 'channels_last' else 1
  default_bn_params = {
      'axis': axis,
      'momentum': 0.99,
      'epsilon': 2e-5,
      'center': True,
      'scale': True,
  }
  default_bn_params.update(params)
  return default_bn_params


# -------------------------------------------------------------------------
#   Residual blocks
# -------------------------------------------------------------------------

class residual_conv_block(tf.keras.layers.Layer):

  def __init__(self, filters, stage, block, strides=(1, 1), attention=None, cut='pre', **kwargs):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        cut: one of 'pre', 'post'. used to decide where skip connection is taken
    # Returns
        Output tensor for the block.
    """
    super(residual_conv_block, self).__init__(**kwargs)

    if cut!="pre" and cut!="post":
      raise ValueError('Cut type not in ["pre", "post"]')
    self.cut = cut
    self.attention = attention
    # get params and names of layers
    conv_params = get_conv_params()
    bn_params = get_bn_params()
    conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)
    self.bn1 = layers.BatchNormalization(name=bn_name + '1', **bn_params)
    self.act1 = layers.Activation('relu', name=relu_name + '1')
    self.downsample = layers.Conv2D(filters, (1, 1), name=sc_name, strides=strides, **conv_params)
    self.zp1 = layers.ZeroPadding2D(padding=(1, 1))
    self.conv2d1 = layers.Conv2D(filters, (3, 3), strides=strides, name=conv_name + '1', **conv_params)
    self.bn2 = layers.BatchNormalization(name=bn_name + '2', **bn_params)
    self.act2 = layers.Activation('relu', name=relu_name + '2')
    self.zp2 = layers.ZeroPadding2D(padding=(1, 1))
    self.conv2d2 = layers.Conv2D(filters, (3, 3), name=conv_name + '2', **conv_params)
    self.add = layers.Add()

  def call(self, inputs):
    x = self.bn1(inputs)
    x = self.act1(x)
    # defining shortcut connection
    if self.cut == 'pre':
      shortcut = inputs
    elif self.cut == 'post':
      shortcut = self.downsample(x)
    # continue with convolution layers
    x = self.zp1(x)
    x = self.conv2d1(x)
    x = self.bn2(x)
    x = self.act2(x)
    x = self.zp2(x)
    x = self.conv2d2(x)
    # use attention block if defined
    if self.attention is not None:
      x = self.attention(x)
    # add residual connection
    x = self.add([x, shortcut])
    return x


class residual_bottleneck_block(tf.keras.layers.Layer):

  def __init__(self, filters, stage, block, strides=None, attention=None, cut='pre', **kwargs):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        cut: one of 'pre', 'post'. used to decide where skip connection is taken
    # Returns
        Output tensor for the block.
    """
    super(residual_bottleneck_block, self).__init__(**kwargs)
    if cut!="pre" and cut!="post":
      raise ValueError('Cut type not in ["pre", "post"]')
    self.cut = cut
    self.attention = attention
    # get params and names of layers
    conv_params = get_conv_params()
    bn_params = get_bn_params()
    conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)
    self.bn1 = layers.BatchNormalization(name=bn_name + '1', **bn_params)
    self.act1 = layers.Activation('relu', name=relu_name + '1')
    self.downsample = layers.Conv2D(filters * 4, (1, 1), name=sc_name, strides=strides, **conv_params)
    self.conv2d1 = layers.Conv2D(filters, (1, 1), name=conv_name + '1', **conv_params)
    self.bn2 = layers.BatchNormalization(name=bn_name + '2', **bn_params)
    self.act2 = layers.Activation('relu', name=relu_name + '2')
    self.zp1 = layers.ZeroPadding2D(padding=(1, 1))
    self.conv2d2 = layers.Conv2D(filters, (3, 3), strides=strides, name=conv_name + '2', **conv_params)
    self.bn3 = layers.BatchNormalization(name=bn_name + '3', **bn_params)
    self.act3 = layers.Activation('relu', name=relu_name + '3')
    self.conv2d3 = layers.Conv2D(filters * 4, (1, 1), name=conv_name + '3', **conv_params)
    self.add = layers.Add()

  def call(self, inputs):
    x = self.bn1(inputs)
    x = self.act1(x)
    # defining shortcut connection
    if cut == 'pre':
      shortcut = inputs
    elif cut == 'post':
      shortcut = self.downsample(x)
    # continue with convolution layers
    x = self.conv2d1(x)
    x = self.bn2(x)
    x = self.act2(x)
    x = self.zp1(x)
    x = self.conv2d2(x)
    x = self.bn3(x)
    x = self.act3(x)
    x = self.conv2d3(x)
    # use attention block if defined
    if self.attention is not None:
      x = self.attention(x)
    # add residual connection
    x = self.add([x, shortcut])
    return x



# -------------------------------------------------------------------------
#   Residual Model Builder
# -------------------------------------------------------------------------


class ResNet(tf.keras.Model):
  def __init__(self, model_params, input_shape=None, input_tensor=None, include_top=True,
               classes=1000, weights='imagenet', **kwargs):
    """Instantiates the ResNet, SEResNet architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    Args:
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    Returns:
        A Keras model instance.
    Raises:
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    super(ResNet, self).__init__(**kwargs)
    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)
    self.model_params = model_params
    # choose residual block type
    ResidualBlock = model_params.residual_block
    if model_params.attention:
      Attention = model_params.attention(**kwargs)
    else:
      Attention = None
    # get parameters for model layers
    no_scale_bn_params = get_bn_params(scale=False)
    bn_params = get_bn_params()
    conv_params = get_conv_params()
    init_filters = 64
    self.bndata = layers.BatchNormalization(name='bn_data', **no_scale_bn_params)
    self.zpdata = layers.ZeroPadding2D(padding=(3, 3))
    self.conv2d0 = layers.Conv2D(init_filters, (7, 7), strides=(2, 2), name='conv0', **conv_params)
    self.bn0 = layers.BatchNormalization(name='bn0', **bn_params)
    self.act0 = layers.Activation('relu', name='relu0')
    self.zp0 = layers.ZeroPadding2D(padding=(1, 1))
    self.pool0 = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='pooling0')
    bodyblocks = []
    for stage, rep in enumerate(self.model_params.repetitions):
      for block in range(rep):
        filters = init_filters * (2 ** stage)
        # first block of first stage without strides because we have maxpooling before
        if block == 0 and stage == 0:
          bodyblocks += [ResidualBlock(filters, stage, block, strides=(1, 1), cut='post', attention=Attention)]
        elif block == 0:
          bodyblocks += [ResidualBlock(filters, stage, block, strides=(2, 2), cut='post', attention=Attention)]
        else:
          bodyblocks += [ResidualBlock(filters, stage, block, strides=(1, 1), cut='pre', attention=Attention)]
    self.bodyblocks = bodyblocks
    self.bn1 = layers.BatchNormalization(name='bn1', **bn_params)
    self.act1 = layers.Activation('relu', name='relu1')
    topblocks = []
    if include_top:
      topblocks += [layers.GlobalAveragePooling2D(name='pool1')]
      topblocks += [layers.Dense(classes, name='fc1')]
      topblocks += [layers.Activation('softmax', name='softmax')]
    self.topblocks = topblocks

  def call(self, inputs):
    x = self.bndata(inputs)
    x = self.zpdata(x)
    # Bottom block
    x = self.conv2d0(x)
    x = self.bn0(x)
    x = self.act0(x)
    x = self.zp0(x)
    x = self.pool0(x)
    # Body blocks
    for block in self.bodyblocks:
      x = block(x)
    x = self.bn1(x)
    x = self.act1(x)
    # Top Block
    for block in self.topblocks:
      x = block(x)
    return x

# if weights:
#   if type(weights) == str and os.path.exists(weights):
#     model.load_weights(weights)
#   else:
#     load_model_weights(model, model_params.model_name,
#                        weights, classes, include_top, **kwargs)


# -------------------------------------------------------------------------
#   Residual Models
# -------------------------------------------------------------------------

MODELS_PARAMS = {
  'resnet18': ModelParams('resnet18', (2, 2, 2, 2), residual_conv_block, None),
  'resnet34': ModelParams('resnet34', (3, 4, 6, 3), residual_conv_block, None),
  'resnet50': ModelParams('resnet50', (3, 4, 6, 3), residual_bottleneck_block, None),
  'resnet101': ModelParams('resnet101', (3, 4, 23, 3), residual_bottleneck_block, None),
  'resnet152': ModelParams('resnet152', (3, 8, 36, 3), residual_bottleneck_block, None),
  'seresnet18': ModelParams('seresnet18', (2, 2, 2, 2), residual_conv_block, ChannelSE),
  'seresnet34': ModelParams('seresnet34', (3, 4, 6, 3), residual_conv_block, ChannelSE),
}


def ResNet18(input_shape=None, input_tensor=None, weights=None, classes=1000, include_top=True, **kwargs):
  return ResNet(
    MODELS_PARAMS['resnet18'],
    input_shape=input_shape,
    input_tensor=input_tensor,
    include_top=include_top,
    classes=classes,
    weights=weights,
    **kwargs
  )


def ResNet34(input_shape=None, input_tensor=None, weights=None, classes=1000, include_top=True, **kwargs):
  return ResNet(
    MODELS_PARAMS['resnet34'],
    input_shape=input_shape,
    input_tensor=input_tensor,
    include_top=include_top,
    classes=classes,
    weights=weights,
    **kwargs
  )


def ResNet50(input_shape=None, input_tensor=None, weights=None, classes=1000, include_top=True, **kwargs):
  return ResNet(
    MODELS_PARAMS['resnet50'],
    input_shape=input_shape,
    input_tensor=input_tensor,
    include_top=include_top,
    classes=classes,
    weights=weights,
    **kwargs
  )


def ResNet101(input_shape=None, input_tensor=None, weights=None, classes=1000, include_top=True, **kwargs):
  return ResNet(
    MODELS_PARAMS['resnet101'],
    input_shape=input_shape,
    input_tensor=input_tensor,
    include_top=include_top,
    classes=classes,
    weights=weights,
    **kwargs
  )


def ResNet152(input_shape=None, input_tensor=None, weights=None, classes=1000, include_top=True, **kwargs):
  return ResNet(
    MODELS_PARAMS['resnet152'],
    input_shape=input_shape,
    input_tensor=input_tensor,
    include_top=include_top,
    classes=classes,
    weights=weights,
    **kwargs
  )


def SEResNet18(input_shape=None, input_tensor=None, weights=None, classes=1000, include_top=True, **kwargs):
  return ResNet(
    MODELS_PARAMS['seresnet18'],
    input_shape=input_shape,
    input_tensor=input_tensor,
    include_top=include_top,
    classes=classes,
    weights=weights,
    **kwargs
  )


def SEResNet34(input_shape=None, input_tensor=None, weights=None, classes=1000, include_top=True, **kwargs):
  return ResNet(
    MODELS_PARAMS['seresnet34'],
    input_shape=input_shape,
    input_tensor=input_tensor,
    include_top=include_top,
    classes=classes,
    weights=weights,
    **kwargs
  )


def preprocess_input(x, **kwargs):
  return x


setattr(ResNet18, '__doc__', ResNet.__doc__)
setattr(ResNet34, '__doc__', ResNet.__doc__)
setattr(ResNet50, '__doc__', ResNet.__doc__)
setattr(ResNet101, '__doc__', ResNet.__doc__)
setattr(ResNet152, '__doc__', ResNet.__doc__)
setattr(SEResNet18, '__doc__', ResNet.__doc__)
setattr(SEResNet34, '__doc__', ResNet.__doc__)
