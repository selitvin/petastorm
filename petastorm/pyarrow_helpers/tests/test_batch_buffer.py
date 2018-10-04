import numpy as np
import pyarrow as pa

from petastorm.pyarrow_helpers.batcher import BatchBuffer
from petastorm.pyarrow_helpers.tests.pyarrow_objects import new_record_batch


def test_single_table():
    table_0_10 = pa.Table.from_batches([new_record_batch(range(0, 10))])

    batcher = BatchBuffer(4)
    assert not batcher.has_next_batch()

    batcher.add_table(table_0_10)


    assert batcher.has_next_batch()
    next_batch = batcher.next_batch()

    assert 4 == next_batch.num_rows
    np.testing.assert_equal(next_batch.column(0).data.to_pylist(), list(range(0, 4)))
    np.testing.assert_equal(next_batch.column(1).data.to_pylist(), list(range(0, 4)))


    assert batcher.has_next_batch()
    next_batch = batcher.next_batch()

    assert 4 == next_batch.num_rows
    np.testing.assert_equal(next_batch.column(0).data.to_pylist(), list(range(4, 8)))
    np.testing.assert_equal(next_batch.column(1).data.to_pylist(), list(range(4, 8)))

    assert not batcher.has_next_batch()


def test_small_record_batches_in_a_table():
    batches = [new_record_batch([i]) for i in range(10)]
    table_0_10 = pa.Table.from_batches(batches)

    batcher = BatchBuffer(4)
    batcher.add_table(table_0_10)


    assert batcher.has_next_batch()
    next_batch = batcher.next_batch()

    assert 4 == next_batch.num_rows
    np.testing.assert_equal(next_batch.column(0).data.to_pylist(), list(range(0, 4)))
    np.testing.assert_equal(next_batch.column(1).data.to_pylist(), list(range(0, 4)))


    assert batcher.has_next_batch()
    next_batch = batcher.next_batch()

    assert 4 == next_batch.num_rows
    np.testing.assert_equal(next_batch.column(0).data.to_pylist(), list(range(4, 8)))
    np.testing.assert_equal(next_batch.column(1).data.to_pylist(), list(range(4, 8)))

    assert not batcher.has_next_batch()
