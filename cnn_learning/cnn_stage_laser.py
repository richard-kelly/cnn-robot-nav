from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import json
import argparse
import numpy as np
import tensorflow as tf

# training hyper parameters are in the config file
with open('config.json', 'r') as fp:
    config = json.load(fp=fp)

parser = argparse.ArgumentParser(description='Learn to drive a robot.')
parser.add_argument(
    '--mode',
    default='train',
    choices=['train', 'eval', 'export'],
    help='train: train a model on data in a dir (with eval_data) or single file (80/20 split); \
        eval: evaluate an existing model on a single data file; \
        export: export a trained model for use with ROS.',
    required=True
)
parser.add_argument(
    '--model_dir',
    help='Location of the model to load or create.',
    required=True
)
parser.add_argument(
    '--export_dir',
    help='Location to save model export in "export" mode.',
    required=False
)
parser.add_argument(
    '--train_data',
    default='data/train/',
    help='Directory or single file to train from. Required for "train" mode.',
    required=False
)
parser.add_argument(
    '--eval_data',
    default='data/eval/',
    help='Path to eval file or dir (will use first file). Required for "train" on dir and "eval" modes.',
    required=False
)

args = parser.parse_args()


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layers
    laser_input_layer = tf.reshape(features["laser"], [-1, 1, 1080, 1])
    goal_input_layer = tf.reshape(features["goal"], [-1, 3])

    # variable to the l1 regularizer
    reg_constant = 0.01

    batch1 = tf.layers.batch_normalization(
        laser_input_layer,
        axis=-1,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )

    conv1 = tf.layers.conv2d(
        inputs=batch1,
        filters=64,
        kernel_size=[1, 7],
        padding="same",
        strides=(1, 3),
        activation=tf.nn.relu,
        kernel_regularizer=tf.contrib.layers.l1_regularizer(scale=config['regularization_scale'])
    )

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[1, 3], strides=(1, 3))

    batch2 = tf.layers.batch_normalization(
        pool1,
        axis=-1,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )

    conv2 = tf.layers.conv2d(
        inputs=batch2,
        filters=64,
        kernel_size=[1, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_regularizer=tf.contrib.layers.l1_regularizer(scale=config['regularization_scale'])
    )

    batch3 = tf.layers.batch_normalization(
        conv2,
        axis=-1,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )

    conv3 = tf.layers.conv2d(
        inputs=batch3,
        filters=64,
        kernel_size=[1, 3],
        padding="same",
        activation=None,
        kernel_regularizer=tf.contrib.layers.l1_regularizer(scale=config['regularization_scale'])
    )

    res1 = pool1 + conv3
    relu1 = tf.nn.relu(res1)

    batch4 = tf.layers.batch_normalization(
        relu1,
        axis=-1,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )

    conv4 = tf.layers.conv2d(
        inputs=batch4,
        filters=64,
        kernel_size=[1, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_regularizer=tf.contrib.layers.l1_regularizer(scale=config['regularization_scale'])
    )

    batch5 = tf.layers.batch_normalization(
        conv4,
        axis=-1,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )

    conv5 = tf.layers.conv2d(
        inputs=batch5,
        filters=64,
        kernel_size=[1, 3],
        padding="same",
        activation=None,
        kernel_regularizer=tf.contrib.layers.l1_regularizer(scale=config['regularization_scale'])
    )

    res2 = conv3 + conv5
    relu2 = tf.nn.relu(res2)

    pool2 = tf.layers.average_pooling2d(inputs=relu2, pool_size=[1, 3], strides=(1, 3))

    # Dense Layer
    # first conv layer and both pooling layers have stride 3, each divides size by 3
    # 1080 -> 360 -> 120 -> 40 * 64 filters/channels
    pool2_flat = tf.reshape(pool2, [-1, 40 * 64])

    combined = tf.concat([pool2_flat, goal_input_layer], 1)
    dense1 = tf.layers.dense(
        inputs=combined,
        units=1024,
        activation=tf.nn.relu,
        kernel_regularizer=tf.contrib.layers.l1_regularizer(scale=config['regularization_scale'])
    )
    dense2 = tf.layers.dense(
        inputs=dense1,
        units=1024,
        activation=tf.nn.relu,
        kernel_regularizer=tf.contrib.layers.l1_regularizer(scale=config['regularization_scale'])
    )
    dense3 = tf.layers.dense(
        inputs=dense2,
        units=512,
        activation=tf.nn.relu,
        kernel_regularizer=tf.contrib.layers.l1_regularizer(scale=config['regularization_scale'])
    )

    # output layer
    output = tf.layers.dense(inputs=dense3, units=2)

    # Updates running average of batch mean and variance for inference time
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # in PREDICT mode return the prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {'predict_output': tf.estimator.export.PredictOutput({"cmd_vel": output})}
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"cmd_vel": output},
            export_outputs=export_outputs
        )

    # Calculate Loss (for both TRAIN and EVAL modes)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    normal_loss = tf.losses.absolute_difference(labels=labels, predictions=output)
    loss = normal_loss + reg_constant * sum(reg_losses)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=config['learning_rate'])
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step()
            )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "trans_error": tf.metrics.mean_absolute_error(
            labels=labels[:, 0], predictions=output[:, 0]),
        "rot_error": tf.metrics.mean_absolute_error(
            labels=labels[:, 1], predictions=output[:, 1])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=normal_loss, eval_metric_ops=eval_metric_ops)


def train_eval_multiple_files(model_dir, train_dir, eval_dir):
    # Create the Estimator
    stage_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=model_dir)

    for _ in range(config['epochs']):
        for filename in os.listdir(train_dir):
            if filename.endswith('.csv'):
                data = np.loadtxt(train_dir + filename, delimiter=',', dtype=np.float32)
                print(filename, ' loaded!')

                train_laser = data[:, 1:1081]
                train_goal = data[:, 1081:1084]
                train_labels = data[:, 1084:]

                train_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"laser": train_laser, "goal": train_goal},
                    y=train_labels,
                    batch_size=config['batch_size'],
                    num_epochs=1,
                    shuffle=True)

                stage_classifier.train(
                    input_fn=train_input_fn,
                    steps=None
                )

        # we only expect one file in eval dir, or it can be a path to a file
        if os.path.isdir(eval_dir):
            eval_file = eval_dir + os.listdir(eval_dir)[0]
        else:
            eval_file = eval_dir
        data = np.loadtxt(eval_file, delimiter=',', dtype=np.float32)

        eval_laser = data[:, 1:1081]
        eval_goal = data[:, 1081:1084]
        eval_labels = data[:, 1084:]

        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"laser": eval_laser, "goal": eval_goal},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)
        eval_results = stage_classifier.evaluate(input_fn=eval_input_fn)
        print('Eval results:')
        print(eval_results)


def train_eval_one_file(model_dir, filename):
    # Create the Estimator
    stage_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=model_dir)

    data = np.loadtxt(filename, delimiter=',', dtype=np.float32)
    print(filename, ' loaded!')

    num_records = data.shape[0]
    cut = math.floor(num_records * 0.8)

    train_laser = data[0:cut, 1:1081]
    train_goal = data[0:cut, 1081:1084]
    train_labels = data[0:cut, 1084:]
    eval_laser = data[cut:, 1:1081]
    eval_goal = data[cut:, 1081:1084]
    eval_labels = data[cut:, 1084:]

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"laser": train_laser, "goal": train_goal},
        y=train_labels,
        batch_size=config['batch_size'],
        num_epochs=config['epochs'],
        shuffle=True)

    stage_classifier.train(
        input_fn=train_input_fn,
        steps=None
    )

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"laser": eval_laser, "goal": eval_goal},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = stage_classifier.evaluate(input_fn=eval_input_fn)
    print('Eval results:')
    print(eval_results)


def eval_one_file(model_dir, filename):
    # Create the Estimator
    stage_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=model_dir)

    data = np.loadtxt(filename, delimiter=',', dtype=np.float32)
    print(filename, ' loaded!')

    eval_laser = data[:, 1:1081]
    eval_goal = data[:, 1081:1084]
    eval_labels = data[:, 1084:]

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"laser": eval_laser, "goal": eval_goal},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = stage_classifier.evaluate(input_fn=eval_input_fn)
    print('Eval results:')
    print(eval_results)


def serving_input_receiver_fn():
    inputs = {
        "laser": tf.placeholder(tf.float32, [None]),
        "goal": tf.placeholder(tf.float32, [None])
    }
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


def save_model(model_dir, export_dir):
    stage_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=model_dir)

    dest = stage_classifier.export_savedmodel(
        export_dir,
        serving_input_receiver_fn
    )

    print("Model saved to ", dest)


def main(unsused_argv):
    mode = args.mode
    print(args.train_data)
    if mode == 'train':
        if os.path.isdir(args.train_data):
            train_eval_multiple_files(args.model_dir, args.train_data, args.eval_data)
        else:
            train_eval_one_file(args.model_dir, args.train_data)
    elif mode == 'eval':
        eval_one_file(args.model_dir, args.eval_data)
    elif mode == 'export':
        save_model(args.model_dir, args.export_dir)


if __name__ == "__main__":
    tf.app.run()
