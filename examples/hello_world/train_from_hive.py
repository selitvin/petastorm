import tensorflow as tf
from tensorflow import data
from datetime import datetime
from tensorflow.python.feature_column import feature_column

from petastorm.reader import Reader
from petastorm.tf_utils import tf_tensors, make_petastorm_dataset

# import horovod.tensorflow as hvd
# hvd.init()
from petastorm.workers_pool.thread_pool import ThreadPool

DATASET_URL = "file:///home/olcay/Uber/bigeta/data/"
TRAIN_URL = DATASET_URL + "train/"
TEST_URL = DATASET_URL + "test/"

TARGET = "ata_sec"

INPUT_FEATURES = ['o8', 'd8', 'o9', 'd9', 'o10', 'd10', 't']


def feature_hash(key_or_keylist, buckets=1000000, dimension=10):
    if isinstance(key_or_keylist, list):
        hash_column = feature_column.crossed_column(key_or_keylist, hash_bucket_size=buckets)
    else:
        hash_column = feature_column.categorical_column_with_hash_bucket(key_or_keylist, hash_bucket_size=buckets)
    embedding_column = feature_column.embedding_column(hash_column, dimension)

    return (hash_column, embedding_column)


def make_feature_columns(buckets, dimension):
    origins = ['o8', 'o9', 'o10']
    destinations = ['d8', 'd9', 'd10']

    wide = []
    deep = []

    for o in origins:
        w, d = feature_hash([o, 't'], buckets=buckets, dimension=dimension)
        wide.append(w)
        deep.append(d)

    for d in destinations:
        w, d = feature_hash([d, 't'], buckets=buckets, dimension=dimension)
        wide.append(w)
        deep.append(d)

    for o in origins:
        for d in destinations:
            w, d = feature_hash([o, d, 't'], buckets=buckets, dimension=dimension)
            wide.append(w)
            deep.append(d)

    return wide, deep


def make_input_fn(reader, feature_list, target_name):
    iterator = reader.next

    def _input_fn():
        batch = iterator()._asdict()
        features = {feature: batch[feature] for feature in feature_list}
        target = batch[target_name]
        return features, target

    return _input_fn


def make_dataset_input_fn(reader, feature_list, target):

    dataset_gen = lambda: make_petastorm_dataset(reader).map(lambda x: ({feature: x._asdict()[feature] for feature in feature_list},
                                                            x._asdict()[target]))

    return dataset_gen


def make_estimator(wide_feature_columns, deep_feature_columns, hparams, run_config):
    estimator = tf.estimator.DNNLinearCombinedRegressor(
        dnn_feature_columns=deep_feature_columns,
        linear_feature_columns=wide_feature_columns,
        dnn_hidden_units=hparams.hidden_units,
        config=run_config)
    return estimator


if __name__ == "__main__":
    run_config = tf.estimator.RunConfig(
        tf_random_seed=19820224,
        model_dir="./trained_models"
    )

    hparams = tf.contrib.training.HParams(
        hash_buckets=10000000,
        embedding_dimension=20,
        hidden_units=[100, 60, 20],
        batch_size=100)

    wide, deep = make_feature_columns(buckets=hparams.hash_buckets, dimension=hparams.embedding_dimension)
    estimator = make_estimator(wide, deep, hparams, run_config)

    stopping_hook = tf.contrib.estimator.stop_if_no_decrease_hook(estimator, "loss", 10000)

    with Reader('file:///home/yevgeni/temp/trueta_tf2_10/', batch_size=hparams.batch_size, reader_pool=ThreadPool(1)) as train_reader:
        # import IPython

        # IPython.embed()
        with Reader('file:///home/yevgeni/temp/trueta_tf2_10/', batch_size=hparams.batch_size, reader_pool=ThreadPool(1)) as test_reader:
            train_input_fn = make_dataset_input_fn(train_reader, INPUT_FEATURES, TARGET)
            test_input_fn = make_dataset_input_fn(test_reader, INPUT_FEATURES, TARGET)

            serving_function = tf.estimator.export.build_raw_serving_input_receiver_fn(
                {name: tf.placeholder(dtype=tf.string, shape=[None], name=name) for name in INPUT_FEATURES}
            )

            train_spec = tf.estimator.TrainSpec(train_input_fn)
            eval_spec = tf.estimator.EvalSpec(
                input_fn=test_input_fn,
                exporters=[tf.estimator.LatestExporter(name="estimator",
                                                       serving_input_receiver_fn=serving_function,
                                                       exports_to_keep=1,
                                                       as_text=True)],
                steps=None,
                throttle_secs=30)

            tf.logging.set_verbosity(tf.logging.INFO)

            start_time = datetime.utcnow()
            print("Started at {}".format(start_time.strftime("%H:%M:%S")))
            print("-------------------------")

            tf.estimator.train_and_evaluate(
                estimator=estimator,
                train_spec=train_spec,
                eval_spec=eval_spec
            )

            end_time = datetime.utcnow()
            elapsed_time = end_time - start_time
            print("Experiment elapsed time: {} seconds".format(elapsed_time.total_seconds()))
