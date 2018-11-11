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
from decimal import Decimal

import pyarrow
from pyarrow import register_default_serialization_handlers


class ArrowTableSerializer(object):

    def serialize(self, rows):
        if rows is not None:
            sink = pyarrow.BufferOutputStream()
            writer = pyarrow.RecordBatchStreamWriter(sink, rows.schema)
            writer.write_table(rows)
            return sink.getvalue()
        else:
            return bytearray([])

    def deserialize(self, serialized_rows):
        reader = pyarrow.open_stream(serialized_rows)
        table = reader.read_all()
        return table
