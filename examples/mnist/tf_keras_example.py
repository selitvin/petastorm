#  Copyright (c) 2017-2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###
# Adapted to petastorm dataset using original contents from
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_softmax.py
###

from __future__ import division, print_function

import argparse
import os

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential

from examples.mnist import DEFAULT_MNIST_DATA_PATH
from petastorm.reader import Reader
from petastorm.tf_utils import make_petastorm_dataset


def train_and_test(dataset_url, training_iterations, batch_size, evaluation_interval):
    batch_size = 128
    num_classes = 10
    epochs = 12

    with Reader(os.path.join(dataset_url, 'train'), num_epochs=epochs) as train_reader:
        with Reader(os.path.join(dataset_url, 'test'), num_epochs=epochs) as test_reader:
            train_dataset = make_petastorm_dataset(train_reader) \
                .map(lambda x: (tf.reshape(x.image, (28, 28, 1)), tf.reshape(x.digit, [1]))) \
                .batch(batch_size, drop_remainder=True)

            test_dataset = make_petastorm_dataset(test_reader) \
                .map(lambda x: (tf.reshape(x.image, (28, 28, 1)), tf.reshape(x.digit, [1]))) \
                .batch(batch_size, drop_remainder=True)

            model = Sequential()
            model.add(Conv2D(32, kernel_size=(3, 3),
                             activation='relu',
                             input_shape=(28, 28, 1)))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(num_classes, activation='softmax'))

            model.compile(optimizer=tf.train.AdamOptimizer(),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            model.fit(train_dataset,
                      verbose=1,
                      epochs=1,
                      steps_per_epoch=100,
                      validation_steps=10,
                      validation_data=test_dataset)

            score = model.evaluate(test_dataset, steps=10, verbose=0)
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Petastorm Tensorflow MNIST Example')
    default_dataset_url = 'file://{}'.format(DEFAULT_MNIST_DATA_PATH)
    parser.add_argument('--dataset-url', type=str,
                        default=default_dataset_url, metavar='S',
                        help='hdfs:// or file:/// URL to the MNIST petastorm dataset'
                             '(default: %s)' % default_dataset_url)
    parser.add_argument('--training-iterations', type=int, default=100, metavar='N',
                        help='number of training iterations to train (default: 100)')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--evaluation-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before evaluating the model accuracy (default: 10)')
    args = parser.parse_args()

    train_and_test(
        dataset_url=args.dataset_url,
        training_iterations=args.training_iterations,
        batch_size=args.batch_size,
        evaluation_interval=args.evaluation_interval,
    )


if __name__ == '__main__':
    main()
