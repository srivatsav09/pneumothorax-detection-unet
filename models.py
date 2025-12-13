from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow as tf


def attention_gate(x, g, inter_channels, name=None):
    """
    Attention gate for U-Net++. Helps the model focus on relevant features.

    Parameters:
    x: skip connection from encoder
    g: gating signal from decoder (lower resolution)
    inter_channels: intermediate channel dimensions
    name: optional name prefix for layers
    """
    theta_x = layers.Conv2D(inter_channels, (1, 1), strides=(1, 1), padding='same', name=f'{name}_theta_x' if name else None)(x)
    phi_g = layers.Conv2D(inter_channels, (1, 1), strides=(1, 1), padding='same', name=f'{name}_phi_g' if name else None)(g)

    add_xg = layers.Add(name=f'{name}_add' if name else None)([theta_x, phi_g])
    act_xg = layers.Activation('relu', name=f'{name}_relu' if name else None)(add_xg)

    psi = layers.Conv2D(1, (1, 1), strides=(1, 1), padding='same', name=f'{name}_psi' if name else None)(act_xg)
    sigmoid_xg = layers.Activation('sigmoid', name=f'{name}_sigmoid' if name else None)(psi)

    upsample_psi = layers.UpSampling2D(size=(2, 2), name=f'{name}_upsample' if name else None)(sigmoid_xg) if tf.keras.backend.int_shape(x)[1] != tf.keras.backend.int_shape(sigmoid_xg)[1] else sigmoid_xg

    y = layers.Multiply(name=f'{name}_multiply' if name else None)([upsample_psi, x])

    return y


def down_conv_block(m, filter_mult, filters, kernel_size, name=None):
    m = layers.Conv2D(filter_mult * filters, kernel_size, padding='same', activation='relu')(m)
    m = layers.BatchNormalization()(m)

    m = layers.Conv2D(filter_mult * filters, kernel_size, padding='same', activation='relu')(m)
    m = layers.BatchNormalization(name=name)(m)

    return m


def up_conv_block(m, prev, filter_mult, filters, kernel_size, prev_2=None, prev_3=None, prev_4=None, name=None, use_attention=False):
    m = layers.Conv2DTranspose(filter_mult * filters, kernel_size, strides=(2, 2), padding='same', activation='relu')(m)
    m = layers.BatchNormalization()(m)

    # Apply attention gates to skip connections if enabled
    if use_attention:
        inter_channels = filter_mult * filters
        prev = attention_gate(prev, m, inter_channels, name=f'{name}_att' if name else None)

    # Concatenate layers; varies between UNet and UNet++
    if prev_4 is not None:
        m = layers.Concatenate()([m, prev, prev_2, prev_3, prev_4])
    elif prev_3 is not None:
        m = layers.Concatenate()([m, prev, prev_2, prev_3])
    elif prev_2 is not None:
        m = layers.Concatenate()([m, prev, prev_2])
    else:
        m = layers.Concatenate()([m, prev])

    m = layers.Conv2D(filter_mult * filters, kernel_size, padding='same', activation='relu')(m)
    m = layers.BatchNormalization(name=name)(m)

    return m


def build_unet(model_input, filters, kernel_size):
    # Downsampling / encoding portion
    conv0 = down_conv_block(model_input, 1, filters, kernel_size)
    pool0 = layers.MaxPooling2D((2, 2))(conv0)

    conv1 = down_conv_block(pool0, 2, filters, kernel_size)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)

    conv2 = down_conv_block(pool1, 4, filters, kernel_size)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)

    conv3 = down_conv_block(pool2, 8, filters, kernel_size)
    pool4 = layers.MaxPooling2D((2, 2))(conv3)

    # Middle of network
    conv4 = down_conv_block(pool4, 16, filters, kernel_size)

    # Upsampling / decoding portion
    uconv3 = up_conv_block(conv4, conv3, 8, filters, kernel_size)

    uconv2 = up_conv_block(uconv3, conv2, 4, filters, kernel_size)

    uconv1 = up_conv_block(uconv2, conv1, 2, filters, kernel_size)

    uconv0 = up_conv_block(uconv1, conv0, 1, filters, kernel_size)

    return uconv0


def build_unet_plus_plus(model_input, filters, kernel_size, l, use_attention=False):
    # Variables names follow the UNet++ paper: [successively downsampled layers_successively upsampled layers)
    # First stage of backbone: downsampling
    conv0_0 = down_conv_block(model_input, 1, filters, kernel_size, name='conv0_0')
    pool0_0 = layers.MaxPooling2D((2, 2))(conv0_0)
    conv1_0 = down_conv_block(pool0_0, 2, filters, kernel_size, name='conv1_0')

    if l > 1:
        # Second stage
        pool1_0 = layers.MaxPooling2D((2, 2))(conv1_0)
        conv2_0 = down_conv_block(pool1_0, 4, filters, kernel_size, name='conv2_0')

        if l > 2:
            # Third stage
            pool2_0 = layers.MaxPooling2D((2, 2))(conv2_0)
            conv3_0 = down_conv_block(pool2_0, 8, filters, kernel_size, name='conv3_0')

            if l > 3:
                # Fourth stage
                pool3_0 = layers.MaxPooling2D((2, 2))(conv3_0)
                conv4_0 = down_conv_block(pool3_0, 16, filters, kernel_size, name='conv4_0')

    # First stage of upsampling and skip connections
    conv0_1 = up_conv_block(conv1_0, conv0_0, 1, filters, kernel_size, name='conv0_1', use_attention=use_attention)
    out = conv0_1

    if l > 1:
        # Second stage
        conv1_1 = up_conv_block(conv2_0, conv1_0, 2, filters, kernel_size, name='conv1_1', use_attention=use_attention)
        conv0_2 = up_conv_block(conv1_1, conv0_1, 1, filters, kernel_size, prev_2=conv0_0, name='conv0_2', use_attention=use_attention)
        out = conv0_2

        if l > 2:
            # Third stage
            conv2_1 = up_conv_block(conv3_0, conv2_0, 4, filters, kernel_size, name='conv2_1', use_attention=use_attention)
            conv1_2 = up_conv_block(conv2_1, conv1_1, 2, filters, kernel_size, prev_2=conv1_0, name='conv1_2', use_attention=use_attention)

            conv0_3 = up_conv_block(conv1_2, conv0_2, 1, filters, kernel_size, prev_2=conv0_1, prev_3=conv0_0,
                                    name='conv0_3', use_attention=use_attention)
            out = conv0_3

            if l > 3:
                # Fourth stage
                conv3_1 = up_conv_block(conv4_0, conv3_0, 8, filters, kernel_size, name='conv3_1', use_attention=use_attention)
                conv2_2 = up_conv_block(conv3_1, conv2_1, 4, filters, kernel_size, prev_2=conv2_0, name='conv2_2', use_attention=use_attention)
                conv1_3 = up_conv_block(conv2_2, conv1_2, 2, filters, kernel_size, prev_2=conv1_1, prev_3=conv1_0,
                                        name='conv1_3', use_attention=use_attention)
                conv0_4 = up_conv_block(conv1_3, conv0_3, 1, filters, kernel_size, prev_2=conv0_2, prev_3=conv0_1,
                                        prev_4=conv0_0, name='conv0_4', use_attention=use_attention)
                out = conv0_4

    return out


def create_segmentation_model(input_size, architecture='unet_plus_plus', l=3, use_attention=False):
    """
    Create a new segmentation model.

    Parameters:
    input_size: int:
        the input size to the segmentation model in pixels. I used 512
    architecture: string:
        'unet' or 'unet_plus_plus' or 'attention_unet_plus_plus' are acceptable arguments.
        UNet++ follows the UNet++ paper: see more details at https://arxiv.org/pdf/1912.05074.pdf
        Attention U-Net++ adds attention gates to focus on relevant features
    l:
        UNet depth; the maximal number of down-convolution and up-convolution blocks
    use_attention: bool:
        whether to use attention gates in U-Net++ (recommended for better performance)
    """
    model_input = layers.Input((input_size, input_size, 1))

    assert l in range(1, 5), f'UNet++ depth {l} not allowed. l must be in range: 1, 2, 3, 4.'

    if architecture == 'unet_plus_plus' or architecture == 'attention_unet_plus_plus':
        if architecture == 'attention_unet_plus_plus':
            use_attention = True
        model_output = build_unet_plus_plus(model_input, 32, (3, 3), l, use_attention=use_attention)
    elif architecture == 'unet':
        model_output = build_unet(model_input, 32, kernel_size=(3, 3))
    else:
        raise AttributeError(f'Network architecture {architecture} does not exist.')

    # Finally - the output sigmoid layer
    output_layer = layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid', name='output_conv')(model_output)

    model = models.Model(inputs=model_input, outputs=output_layer)
    model.summary()

    return model


def create_classification_model(input_size, bb='EfficientNetB3'):
    """
    Create a classification model by loading pretrained ImageNet weights.
    Supports DenseNet, VGG16, and EfficientNet backbones (EfficientNet recommended for best performance).

    Parameters:
    input_size: int:
        the input size to the classification model in pixels. I used 512
    bb: string:
        the backbone to use for image classification. Acceptable arguments are:
        DenseNet121, DenseNet169, DenseNet201, VGG16, EfficientNetB0, EfficientNetB3 (recommended), EfficientNetB4
    """

    assert bb in ['DenseNet121', 'DenseNet169', 'DenseNet201', 'VGG16',
                  'EfficientNetB0', 'EfficientNetB3', 'EfficientNetB4'], \
        f'Backbone {bb} not in list: DenseNet121, DenseNet169, DenseNet201, VGG16, EfficientNetB0, EfficientNetB3, EfficientNetB4'

    if bb == 'DenseNet201':
        pretrained_model = applications.DenseNet201(include_top=False,
                                                    weights='imagenet',
                                                    input_shape=(input_size, input_size, 3),
                                                    pooling=None)
    elif bb == 'DenseNet169':
        pretrained_model = applications.DenseNet169(include_top=False,
                                                    weights='imagenet',
                                                    input_shape=(input_size, input_size, 3),
                                                    pooling=None)
    elif bb == 'DenseNet121':
        pretrained_model = applications.DenseNet121(include_top=False,
                                                    weights='imagenet',
                                                    input_shape=(input_size, input_size, 3),
                                                    pooling=None)
    elif bb == 'VGG16':
        pretrained_model = applications.VGG16(include_top=False,
                                              weights='imagenet',
                                              input_shape=(input_size, input_size, 3),
                                              pooling=None)
    elif bb == 'EfficientNetB0':
        pretrained_model = applications.EfficientNetB0(include_top=False,
                                                       weights='imagenet',
                                                       input_shape=(input_size, input_size, 3),
                                                       pooling=None)
    elif bb == 'EfficientNetB3':
        pretrained_model = applications.EfficientNetB3(include_top=False,
                                                       weights='imagenet',
                                                       input_shape=(input_size, input_size, 3),
                                                       pooling=None)
    elif bb == 'EfficientNetB4':
        pretrained_model = applications.EfficientNetB4(include_top=False,
                                                       weights='imagenet',
                                                       input_shape=(input_size, input_size, 3),
                                                       pooling=None)

    # Fine-tuning strategy: freeze early layers, unfreeze later layers
    # For EfficientNet, unfreeze top blocks for better medical image adaptation
    if bb.startswith('EfficientNet'):
        # Freeze all layers except the last 20%
        total_layers = len(pretrained_model.layers)
        freeze_until = int(total_layers * 0.8)
        for i, layer in enumerate(pretrained_model.layers):
            if i < freeze_until:
                layer.trainable = False
            else:
                layer.trainable = True
    else:
        # Original DenseNet/VGG freezing logic
        for l in pretrained_model.layers:
            if not l.name.startswith('conv5_block2') or l.name.startswith('conv5_block3'):
                l.trainable = False

    model_output = layers.GlobalAveragePooling2D()(pretrained_model.output)
    model_output = layers.Dropout(0.3)(model_output)  # Add dropout for regularization
    model_output = layers.Dense(44, activation='relu')(model_output)
    model_output = layers.Dropout(0.2)(model_output)
    model_output = layers.Dense(1, activation='sigmoid')(model_output)

    model = models.Model(inputs=pretrained_model.input, outputs=model_output)
    model.summary()

    return model


def unet_pp_pretrain_model(input_size, l=3):
    """
    Create a UNet++ segmentation model with a 1000-way classification head for pretraining on ImageNet. I've pretrained
    an l3 model on ImageNet, you can find a link in the readme file

    Parameters:
    input_size: int:
        the input size to the segmentation model in pixels. I used 512
    l:
        UNet depth; the maximal number of down-convolution and up-convolution blocks
    """
    model_input = layers.Input((input_size, input_size, 1))
    model_output = build_unet_plus_plus(model_input, 32, (3, 3), l)

    imagenet_head = layers.GlobalAveragePooling2D()(model_output)
    imagenet_head = layers.Dense(1000, activation='softmax')(imagenet_head)

    model = models.Model(inputs=model_input, outputs=imagenet_head)
    model.summary()

    return model
