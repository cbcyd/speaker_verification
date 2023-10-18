"""
This script includes open source components from conv_models.py,
developed by @philipperemy in the deep-speaker project.

Refer to the original project here:
https://github.com/philipperemy/deep-speaker/
"""
import os

# Set environment variables to hide tensorflow warnings before importing tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow.keras.backend as K

from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Input, Lambda, Reshape
from tensorflow.keras.models import Model

from modules.audio import NUM_FBANKS, NUM_FRAMES

class DeepSpeakerModel:
    def __init__(self, batch_input_shape=(None, NUM_FRAMES, NUM_FBANKS, 1)):
        """
        Initialize DeepSpeakerModel object.
        """
        self.clipped_relu_count = 0
        inputs = Input(batch_shape=batch_input_shape, name="input")
        x = self.cnn_component(inputs)
        x = Reshape((-1, 2048))(x)

        # Temporal average layer. axis=1 is time.
        x = Lambda(lambda y: K.mean(y, axis=1), name="average")(x)
        x = Dense(512, name="affine")(x)

        # Apply L2 normalization
        x = Lambda(lambda y: K.l2_normalize(y, axis=1), name="ln")(x)
        self.rescnn = Model(inputs, x, name="ResCNN")

    def clipped_relu(self, inputs):
        """
        Create a clipped ReLU layer with the given inputs, where ReLU range is [0, 20].
        """
        relu = Lambda(
            lambda y: K.minimum(K.maximum(y, 0), 20),
            name=f"clipped_relu_{self.clipped_relu_count}",
        )(inputs)
        self.clipped_relu_count += 1
        return relu

    def identity_block(self, input_tensor, kernel_size, filters, stage, block):
        """
        Create a identity block with given parameters.
        """
        conv_name_base = f"res{stage}_{block}_branch"

        x = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            activation=None,
            padding="same",
            kernel_initializer="glorot_uniform",
            kernel_regularizer=regularizers.l2(l=0.0001),
            name=conv_name_base + "_2a",
        )(input_tensor)

        x = BatchNormalization(name=conv_name_base + "_2a_bn")(x)
        x = self.clipped_relu(x)

        x = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            activation=None,
            padding="same",
            kernel_initializer="glorot_uniform",
            kernel_regularizer=regularizers.l2(l=0.0001),
            name=conv_name_base + "_2b",
        )(x)

        x = BatchNormalization(name=conv_name_base + "_2b_bn")(x)
        x = self.clipped_relu(x)

        x = layers.add([x, input_tensor])
        x = self.clipped_relu(x)
        return x

    def conv_and_res_block(self, inp, filters, stage):
        """
        Create a block of convolution and residual layers with the given parameters.
        """
        conv_name = "conv{}-s".format(filters)
        o = Conv2D(
            filters=filters,
            kernel_size=5,
            strides=2,
            activation=None,
            padding="same",
            kernel_initializer="glorot_uniform",
            kernel_regularizer=regularizers.l2(l=0.0001),
            name=conv_name,
        )(inp)

        o = BatchNormalization(name=conv_name + "_bn")(o)
        o = self.clipped_relu(o)

        for i in range(3):
            o = self.identity_block(
                o, kernel_size=3, filters=filters, stage=stage, block=i
            )
        return o

    def cnn_component(self, inp):
        """
        Create a CNN component from input tensor by using a series of convolution and residual blocks.
        """
        x = self.conv_and_res_block(inp, 64, stage=1)
        x = self.conv_and_res_block(x, 128, stage=2)
        x = self.conv_and_res_block(x, 256, stage=3)
        x = self.conv_and_res_block(x, 512, stage=4)
        return x

    def set_weights(self, w):
        """
        Set the weights for each layer in the model based on the weights value 'w'.
        """
        for layer, layer_w in zip(self.m.layers, w):
            layer.set_weights(layer_w)