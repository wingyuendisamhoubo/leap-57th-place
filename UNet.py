import os

os.environ["KERAS_BACKEND"] = "jax"

import gc
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

from keras.regularizers import l2

import tensorflow as tf
import jax
import keras
from keras.layers import Conv1D, GroupNormalization, Activation

from sklearn import metrics

from tqdm.notebook import tqdm

print(tf.__version__)
print(jax.__version__)

# Transformer block
@keras.saving.register_keras_serializable()
def transformer_block(inputs, head_size, num_heads, ff_dim, dropout=0.2):
    attention_output = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    attention_output = tf.keras.layers.Dropout(dropout)(attention_output)
    attention_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention_output + inputs)

    ff_output = tf.keras.layers.Dense(ff_dim, activation='gelu')(attention_output)
    ff_output = tf.keras.layers.Dropout(dropout)(ff_output)
    ff_output = tf.keras.layers.Dense(inputs.shape[-1])(ff_output)
    ff_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(ff_output + attention_output)

    return ff_output

# Transformer_CA block
@keras.saving.register_keras_serializable()
def transformer_CA_block(inputs, skip_features, head_size, num_heads, ff_dim, dropout=0.2):
    attention_output = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(skip_features, inputs)
    attention_output = tf.keras.layers.Dropout(dropout)(attention_output)
    attention_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention_output + skip_features)

    ff_output = tf.keras.layers.Dense(ff_dim, activation='gelu')(attention_output)
    ff_output = tf.keras.layers.Dropout(dropout)(ff_output)
    ff_output = tf.keras.layers.Dense(inputs.shape[-1])(ff_output)
    ff_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(ff_output + attention_output)

    return ff_output

# ResBlock function
@keras.saving.register_keras_serializable()
def res_block(x, filters, output_filters=None, groups=8):
    if output_filters is None:
        output_filters = filters
    norm1 = GroupNormalization(groups=groups, axis=-1)(x)
    silu1 = tf.keras.layers.Activation('swish')(norm1)
    conv1 = tf.keras.layers.Conv1D(filters, kernel_size=3, padding='same')(silu1)

    norm2 = GroupNormalization(groups=groups, axis=-1)(conv1)
    silu2 = tf.keras.layers.Activation('swish')(norm2)
    conv2 = tf.keras.layers.Conv1D(output_filters, kernel_size=3, padding='same')(silu2)

    if x.shape[-1] != conv2.shape[-1]:
        x = tf.keras.layers.Conv1D(output_filters, kernel_size=1, padding='same')(x)
    x = tf.keras.layers.Add()([conv2, x])
    norm3 = GroupNormalization(groups=groups, axis=-1)(x)
    return norm3

# Downsample block
@keras.saving.register_keras_serializable()
def repeat_block(x, filters, repeat):
    for _ in range(repeat):
        x = res_block(x, filters)
    return x

# Upsample block
@keras.saving.register_keras_serializable()
def upsample_block(x, filters, repeat, concat_layer):
    num_heads = filters // 4
    concat_layer = transformer_CA_block(x, concat_layer, head_size=4, num_heads=num_heads, ff_dim=filters*2)
    x = tf.keras.layers.Conv1DTranspose(filters, kernel_size=2, strides=2, padding='same')(x)
    x = tf.keras.layers.Concatenate()([x, concat_layer])
    for _ in range(repeat):
        x = res_block(x, filters)
    return x


keras.utils.clear_session()


def x_to_seq(x):
    x_seq0 = keras.ops.transpose(keras.ops.reshape(x[:, 0:60 * 6], (-1, 6, 60)), (0, 2, 1))
    x_seq1 = keras.ops.transpose(keras.ops.reshape(x[:, 60 * 6 + 16:60 * 9 + 16], (-1, 3, 60)), (0, 2, 1))
    x_flat = keras.ops.reshape(x[:, 60 * 6:60 * 6 + 16], (-1, 1, 16))
    x_flat = keras.ops.repeat(x_flat, 60, axis=1)
    return keras.ops.concatenate([x_seq0, x_seq1, x_flat], axis=-1)

    # Build 1D U-Net model


def create_unet(input_shape):
    inputs = keras.layers.Input(shape=input_shape)

    # Encoder
    encoder_1 = repeat_block(inputs, 128, 2)  # 64 x 128 x 2
    encoder_1 = transformer_block(encoder_1, head_size=16, num_heads=8, ff_dim=256)
    encoder_1_down = keras.layers.MaxPooling1D(pool_size=2, strides=2)(encoder_1)  # Downsample # 32 x 128

    encoder_2 = repeat_block(encoder_1_down, 256, 2)  # 32 x 256 x 2
    encoder_2 = transformer_block(encoder_2, head_size=32, num_heads=8, ff_dim=512)
    encoder_2_down = keras.layers.MaxPooling1D(pool_size=2, strides=2)(encoder_2)  # Downsample # 16 x 256

    encoder_3 = repeat_block(encoder_2_down, 256, 2)  # 16 x 256 x 2
    encoder_3 = transformer_block(encoder_3, head_size=32, num_heads=8, ff_dim=512)
    encoder_3_down = keras.layers.MaxPooling1D(pool_size=2, strides=2)(encoder_3)  # Downsample # 8 x 256

    encoder_4 = repeat_block(encoder_3_down, 256, 2)  # 8 x 256 x 2

    # Bottleneck (Transformer)

    bottleneck = transformer_block(encoder_4, head_size=32, num_heads=8, ff_dim=512)
    bottleneck = repeat_block(bottleneck, 256, 1)  # 8 x 256 x 2

    encoder_4 = transformer_CA_block(bottleneck, encoder_4, head_size=4, num_heads=64, ff_dim=512)
    decoder_1 = keras.layers.Concatenate()([bottleneck, encoder_4])
    decoder_1_block = repeat_block(decoder_1, 256, 3)  # 8 x 256 x 3
    decoder_1_block = transformer_block(decoder_1_block, head_size=32, num_heads=8, ff_dim=512)
    decoder_1_upsample = keras.layers.Conv1DTranspose(256, kernel_size=2, strides=2, padding='same')(
        decoder_1_block)  # Upsample # 16 x 256

    encoder_3 = transformer_CA_block(decoder_1_upsample, encoder_3, head_size=4, num_heads=64, ff_dim=512)
    decoder_2 = keras.layers.Concatenate()([decoder_1_upsample, encoder_3])
    decoder_2_block = repeat_block(decoder_2, 256, 3)  # 16 x 256 x 3
    decoder_2_block = transformer_block(decoder_2_block, head_size=32, num_heads=8, ff_dim=512)
    decoder_2_upsample = keras.layers.Conv1DTranspose(256, kernel_size=2, strides=2, padding='same')(
        decoder_2_block)  # Upsample # 32 x 256

    encoder_2 = transformer_CA_block(decoder_2_upsample, encoder_2, head_size=4, num_heads=64, ff_dim=512)
    decoder_3 = keras.layers.Concatenate()([decoder_2_upsample, encoder_2])
    decoder_3_block = repeat_block(decoder_3, 256, 3)  # 32 x 256 x 3
    decoder_3_block = transformer_block(decoder_3_block, head_size=32, num_heads=8, ff_dim=512)
    decoder_3_upsample = keras.layers.Conv1DTranspose(256, kernel_size=2, strides=2, padding='same')(
        decoder_3_block)  # Upsample # 64 x 256

    decoder_4 = keras.layers.Concatenate()([decoder_3_upsample, encoder_1])
    decoder_4_block = repeat_block(decoder_4, 256, 3)  # 64 x 256 x 3

    model = keras.models.Model(inputs, decoder_4_block)
    return model
