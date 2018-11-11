import numpy as np
import pandas as pd
import pyarrow as pa
from pyspark.sql.types import LongType

from petastorm.codecs import NdarrayCodec, ScalarCodec
from petastorm.pyarrow_helpers.numpy_conversion import batch_table_to_dict_of_numpy
from petastorm.unischema import Unischema, UnischemaField


def test_to_dict():
    schema = Unischema('schema', [
        UnischemaField('a', np.int32, (), ScalarCodec(LongType()), False),
        UnischemaField('b', np.int32, (), ScalarCodec(LongType()), False),
    ])
    table_pandas = pd.DataFrame({'a': [0, 1, 2], 'b': [5, 6, 7]})


    table = pa.Table.from_pandas(table_pandas, preserve_index=False)
    as_numpy = batch_table_to_dict_of_numpy(table, schema)
    assert set(as_numpy.keys()) == {'a', 'b'}
    np.testing.assert_array_equal(as_numpy['a'], [0, 1, 2])
    np.testing.assert_array_equal(as_numpy['b'], [5, 6, 7])
