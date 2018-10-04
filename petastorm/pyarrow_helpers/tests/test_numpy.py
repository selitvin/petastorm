import numpy as np
import pandas as pd
import pyarrow as pa

from petastorm.pyarrow_helpers.numpy import table_to_dict_of_numpy


def test_to_dict():
    table_pandas = pd.DataFrame({'a': [0, 1, 2], 'b': [5, 6, 7]})
    table = pa.Table.from_pandas(table_pandas)
    as_numpy = table_to_dict_of_numpy(table)
    assert set(as_numpy.keys()) == {'a', 'b'}
    np.testing.assert_array_equal(as_numpy['a'], [0, 1, 2])
    np.testing.assert_array_equal(as_numpy['b'], [5, 6, 7])
