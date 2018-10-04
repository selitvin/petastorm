from collections import deque

import pyarrow as pa


class BatchBuffer(object):
    def __init__(self, batch_size):
        self._batch_size = batch_size
        self._buffer = deque()
        self._head_idx = 0
        self._cumulative_len = 0

    def add_record_batch(self, record_batch):
        self._buffer.append(record_batch)
        self._cumulative_len += record_batch.num_rows

    def add_table(self, table):
        record_batches = table.to_batches()
        for record_batch in record_batches:
            self.add_record_batch(record_batch)

    def has_next_batch(self):
        return self._head_idx + self._batch_size < self._cumulative_len

    def next_batch(self):

        assert self.has_next_batch()

        bs = self._batch_size

        result = []
        result_rows = 0
        while result_rows < bs and self._cumulative_len > 0:
            head = self._buffer[0]
            piece = head[self._head_idx:self._head_idx + self._batch_size]
            self._head_idx += piece.num_rows
            result_rows += piece.num_rows
            result.append(piece)

            if head.num_rows == self._head_idx:
                self._head_idx = 0
                self._buffer.popleft()
                self._cumulative_len -= head.num_rows

        return pa.Table.from_batches(result)
