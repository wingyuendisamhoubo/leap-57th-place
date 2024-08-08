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

from UNet import create_unet

BATCH_SIZE = 512

DATA = "autodl-tmp/kaggle/input"
DATA_TFREC = "autodl-tmp/archive"
model_path = "model.keras"

sample = pl.read_csv(os.path.join(DATA, "sample_submission.csv"), n_rows=1)
TARGETS = sample.select(pl.exclude('sample_id')).columns
print(len(TARGETS))

print(len(TARGETS))

def _parse_function(example_proto):
    feature_description = {
        'x': tf.io.FixedLenFeature([556], tf.float32),
        'targets': tf.io.FixedLenFeature([368], tf.float32)
    }
    e = tf.io.parse_single_example(example_proto, feature_description)
    return e['x'], e['targets']


train_files = [os.path.join(DATA_TFREC, "train_%.3d.tfrec" % i) for i in range(100)]
valid_files = [os.path.join(DATA_TFREC, "train_%.3d.tfrec" % i) for i in range(100, 101)]

train_options = tf.data.Options()
train_options.deterministic = True

ds_train = (
    tf.data.Dataset.from_tensor_slices(train_files)
    .with_options(train_options)
    .shuffle(100)
    .interleave(
        lambda file: tf.data.TFRecordDataset(file).map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE),
        num_parallel_calls=tf.data.AUTOTUNE,
        cycle_length=10,
        block_length=1000,
        deterministic=True
    )
    .shuffle(4 * BATCH_SIZE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

ds_valid = (
    tf.data.TFRecordDataset(valid_files)
    .map(_parse_function)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

norm_x = keras.layers.Normalization()
norm_x.adapt(ds_train.map(lambda x, y: x))

plt.scatter(
    norm_x.mean.squeeze(),
    norm_x.variance.squeeze() ** 0.5,
    marker=".",
    alpha=0.5
)
plt.xscale('log')
plt.yscale('log')

norm_y = keras.layers.Normalization()
norm_y.adapt(ds_train.map(lambda x, y: y))

mean_y = norm_y.mean
stdd_y = keras.ops.maximum(1e-20, norm_y.variance ** 0.5)

plt.scatter(
    mean_y.squeeze(),
    stdd_y.squeeze(),
    marker=".",
    alpha=0.5
)
plt.xscale('log')
plt.yscale('log')


@keras.saving.register_keras_serializable(package="MyMetrics", name="ClippedR2Score")
class ClippedR2Score(keras.metrics.Metric):
    def __init__(self, name='r2_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.base_metric = keras.metrics.R2Score(class_aggregation=None)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.base_metric.update_state(y_true, y_pred, sample_weight=None)

    def result(self):
        return keras.ops.mean(keras.ops.clip(self.base_metric.result(), 0.0, 1.0))

    def reset_states(self):
        self.base_metric.reset_states()

epochs = 15
learning_rate = 3e-4

epochs_warmup = 1
epochs_ending = 2
steps_per_epoch = int(np.ceil(len(train_files) * 100_000 / BATCH_SIZE))

lr_scheduler = keras.optimizers.schedules.CosineDecay(
    1e-6,
    (epochs - epochs_warmup - epochs_ending) * steps_per_epoch,
    warmup_target=learning_rate,
    warmup_steps=steps_per_epoch * epochs_warmup,
    alpha=0.1
)

plt.plot([lr_scheduler(it) for it in range(0, epochs * steps_per_epoch, steps_per_epoch)]);

X_input = x = keras.layers.Input(ds_train.element_spec[0].shape[1:])
x = keras.layers.Normalization(mean=norm_x.mean, variance=norm_x.variance)(x)
x = x_to_seq(x)

# Zero-padding at the beginning and end of the sequence to extend the length from 60 to 64
x = keras.layers.ZeroPadding1D(padding=(2, 2))(x)
#
e = keras.layers.Conv1D(48, 1, padding='same')(x)
e = create_unet(e.shape[1:])(e)

# Use a Lambda layer to remove the first and last 2 time steps
e = e[:, 2:-2, :]

p_all = keras.layers.Conv1D(14, 1, padding='same')(e)
print(p_all.shape)

p_seq = p_all[:, :, :6]
p_seq = keras.ops.transpose(p_seq, (0, 2, 1))
p_seq = keras.layers.Flatten()(p_seq)
assert p_seq.shape[-1] == 360

p_flat = p_all[:, :, 6:6 + 8]
p_flat = keras.ops.mean(p_flat, axis=1)
assert p_flat.shape[-1] == 8

P = keras.ops.concatenate([p_seq, p_flat], axis=1)

# Build & compile the model
model = keras.Model(X_input, P)
model.compile(
    loss=keras.losses.Huber(delta=1.0),
    optimizer=keras.optimizers.Adam(lr_scheduler),
    metrics=[keras.metrics.MeanSquaredError(),
             keras.metrics.R2Score(class_aggregation="variance_weighted_average"),
             ]  # Updated R2Score
)
model.build(tuple(ds_train.element_spec[0].shape))
model.summary()



%%time

# Normalize target values for training and validation datasets
ds_train_target_normalized = ds_train.map(lambda x, y: (x, (y - mean_y) / stdd_y))
ds_valid_target_normalized = ds_valid.map(lambda x, y: (x, (y - mean_y) / stdd_y))


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Model checkpoint callback
model_checkpoint = ModelCheckpoint(
    filepath='model_epoch_{epoch:02d}.keras',  # Save model with epoch number
    monitor='val_loss',  # Monitor validation loss
    save_best_only=False,  # Save all models, not just the best one
    save_weights_only=False,  # Save the entire model structure and weights
    mode='min',  # 'min' indicates saving when the monitored value decreases
    verbose=2  # Provide detailed logging
)


    # Train the model
history = model.fit(
    ds_train_target_normalized,  # Training data
    validation_data=ds_valid_target_normalized,  # Validation data
    epochs=epochs,  # Number of epochs to train
    verbose=1 if is_interactive() else 2,  # Verbose output
    callbacks=model_checkpoint  # List of callbacks to apply during training
)

plt.plot(history.history['loss'], color='tab:blue')
plt.plot(history.history['val_loss'], color='tab:red')
plt.yscale('log')




