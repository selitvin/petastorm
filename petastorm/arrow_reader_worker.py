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
from __future__ import division

import hashlib
import operator

import numpy as np
import pyarrow
import pyarrow as pa
from pyarrow import parquet as pq
from pyarrow.parquet import ParquetFile

from petastorm.cache import NullCache
from petastorm.workers_pool.worker_base import WorkerBase


class ArrowReaderWorker(WorkerBase):
    def __init__(self, worker_id, publish_func, args):
        super(ArrowReaderWorker, self).__init__(worker_id, publish_func, args)

        self._filesystem = args[0]
        self._dataset_path = args[1]
        self._schema = args[2]
        self._ngram = args[3]
        self._split_pieces = args[4]
        self._local_cache = args[5]

        # We create datasets lazily in the first invocation of 'def process'. This speeds up startup time since
        # all Worker constructors are serialized
        self._dataset = None

    # pylint: disable=arguments-differ
    def process(self, piece_index, worker_predicate, shuffle_row_drop_partition):
        """Main worker function. Loads and returns all rows matching the predicate from a rowgroup

        Looks up the requested piece (a single row-group in a parquet file). If a predicate is specified,
        columns needed by the predicate are loaded first. If no rows in the rowgroup matches the predicate criteria
        the rest of the columns are not loaded.

        :param piece_index:
        :param shuffle_row_drop_partition: A tuple 2 of the current row drop partition and the total number
            of partitions.
        :return:
        """

        if not self._dataset:
            self._dataset = pq.ParquetDataset(
                self._dataset_path,
                filesystem=self._filesystem,
                validate_schema=False)

        piece = self._split_pieces[piece_index]

        # Create pyarrow file system
        parquet_file = ParquetFile(self._dataset.fs.open(piece.path))

        if not isinstance(self._local_cache, NullCache):
            if worker_predicate:
                raise RuntimeError('Local cache is not supported together with predicates, '
                                   'unless the dataset is partitioned by the column the predicate operates on.')
            if shuffle_row_drop_partition[1] != 1:
                raise RuntimeError('Local cache is not supported together with shuffle_row_drop_partitions > 1')

        if worker_predicate:
            all_cols = self._load_rows_with_predicate(parquet_file, piece, worker_predicate, shuffle_row_drop_partition)
        else:
            # Using hash of the dataset path with the relative path in order to:
            #  1. Make sure if a common cache serves multiple processes (e.g. redis), we don't have conflicts
            #  2. Dataset path is hashed, to make sure we don't create too long keys, which maybe incompatible with
            #     some cache implementations
            #  3. Still leave relative path and the piece_index in plain text to make it easier to debug
            cache_key = '{}:{}:{}'.format(hashlib.md5(self._dataset_path.encode('utf-8')).hexdigest(),
                                          piece.path, piece_index)
            all_cols = self._local_cache.get(cache_key,
                                             lambda: self._load_rows(parquet_file, piece, shuffle_row_drop_partition))

        if self._ngram:
            all_cols = self._ngram.form_ngram(data=all_cols, schema=self._schema)

        if all_cols:
            self.publish_func(all_cols)

    def _load_rows(self, pq_file, piece, shuffle_row_drop_range):
        """Loads all rows from a piece"""

        # pyarrow would fail if we request a column names that the dataset is partitioned by, so we strip them from
        # the `columns` argument.
        partitions = self._dataset.partitions
        column_names_in_schema = set(field.name for field in self._schema.fields.values())
        column_names = column_names_in_schema - partitions.partition_names

        table = self._read_with_shuffle_row_drop(piece, pq_file, column_names, shuffle_row_drop_range)

        table_column_names = {column.name for column in table.columns}

        return self.decode_columns(table, table_column_names.intersection(column_names_in_schema))

    def decode_columns(self, table, column_names):
        output_columns = filter(lambda column: column.name in column_names, table.columns)

        decoded_columns = []

        for column in output_columns:
            unischema_field = self._schema.fields[column.name]
            decoded_columns.append(self._schema.fields[column.name].codec.decode_column(unischema_field, column))

        return pyarrow.Table.from_arrays(decoded_columns)

    def _load_rows_with_predicate(self, pq_file, piece, worker_predicate, shuffle_row_drop_partition):
        """Loads all rows that match a predicate from a piece"""

        # 1. Read all columns needed by predicate and decode
        # 2. Apply the predicate. If nothing matches, exit early
        # 3. Read the remaining columns and decode
        # 4. Combine with columns already decoded for the predicate.

        # Split all column names into ones that are needed by predicateand the rest.
        predicate_column_names = set(worker_predicate.get_fields())

        if not predicate_column_names:
            raise ValueError('At least one field name must be returned by predicate\'s get_field() method')

        all_schema_names = set(field.name for field in self._schema.fields.values())

        invalid_column_names = predicate_column_names - all_schema_names
        if invalid_column_names:
            raise ValueError('At least some column names requested by the predicate ({}) '
                             'are not valid schema names: ({})'.format(', '.join(invalid_column_names),
                                                                       ', '.join(all_schema_names)))

        other_column_names = all_schema_names - predicate_column_names  # - \
        # self._dataset.partitions.partition_names

        # Read columns needed for the predicate
        table_for_predicates = self._read_with_shuffle_row_drop(piece, pq_file, predicate_column_names,
                                                                shuffle_row_drop_partition)

        decoded_table_for_predicates = self.decode_columns(table_for_predicates, predicate_column_names)
        # # Decode values
        #
        # # Use the predicate to filter
        # match_predicate_mask = np.empty((decoded_table_for_predicates.num_rows,), dtype=np.bool_)
        #
        # column_names = map(operator.attrgetter('name'), table_for_predicates.columns)
        # for row_idx in six.moves.range(table_for_predicates.num_rows):
        #     values = [column.data[row_idx] for column in table_for_predicates.columns]
        #     row = dict(zip(column_names, values))
        #     match_predicate_mask[row_idx] = worker_predicate.do_include(row)

        decoded_pandas_for_predicates = decoded_table_for_predicates.to_pandas()
        match_predicate_mask = decoded_pandas_for_predicates.apply(
            lambda series: worker_predicate.do_include(series.to_dict()), axis=1)
        erase_mask = match_predicate_mask.map(operator.not_)

        # Don't have anything left after filtering? Exit early.
        if erase_mask.all():
            return []

        decoded_pandas_for_predicates[erase_mask] = None

        if other_column_names:
            # Read remaining columns
            other_table = self._read_with_shuffle_row_drop(piece, pq_file, other_column_names,
                                                           shuffle_row_drop_partition)
            other_table_decoded = self.decode_columns(other_table, other_column_names)

            other_table_pandas = other_table_decoded.to_pandas()
            other_table_pandas[erase_mask] = None

            for predicate_column in decoded_pandas_for_predicates.columns:
                other_table_pandas[predicate_column] = decoded_pandas_for_predicates[predicate_column]

            result_table_pandas = other_table_pandas
        else:
            result_table_pandas = decoded_pandas_for_predicates

        filtered_result = pa.Table.from_pandas(result_table_pandas[match_predicate_mask], preserve_index=False)
        return filtered_result

    def _read_with_shuffle_row_drop(self, piece, pq_file, column_names, shuffle_row_drop_partition):
        table = piece.read(
            open_file_func=lambda _: pq_file,
            columns=column_names,
            partitions=self._dataset.partitions
        )

        num_rows = len(table)
        num_partitions = shuffle_row_drop_partition[1]
        this_partition = shuffle_row_drop_partition[0]

        if num_partitions > 1:
            data_frame_pandas = table.to_pandas()
            partition_indexes = np.floor(np.arange(num_rows) / (float(num_rows) / min(num_rows, num_partitions)))

            if self._ngram:
                # If we have an ngram we need to take elements from the next partition to build the sequence
                next_partition_indexes = np.where(partition_indexes >= this_partition + 1)
                if next_partition_indexes[0].size:
                    next_partition_to_add = next_partition_indexes[0][0:self._ngram.length - 1]
                    partition_indexes[next_partition_to_add] = this_partition

            table = pyarrow.Table.from_pandas(data_frame_pandas.loc[partition_indexes == this_partition])

        return table  # .to_dict('records')
