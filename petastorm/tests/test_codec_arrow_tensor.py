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
import pickle

import numpy as np
import pytest

from petastorm.codecs import ArrowTensorCodec
from petastorm.unischema import UnischemaField


def test_arrow_tensor_codec():
    codec = ArrowTensorCodec()
    field = UnischemaField(name='field_image', numpy_dtype=np.float64, shape=(2, 3, 4), codec=codec,
                           nullable=False)

    expected = np.random.random((2, 3, 4)).astype(np.float64)
    actual = codec.decode(field, pickle.loads(pickle.dumps(codec.encode(field, expected))))
    np.testing.assert_array_equal(actual, expected)

    with pytest.raises(ValueError):
        codec.encode(field, np.zeros((10, 10, 10, 10), dtype=np.uint8))
