import pyarrow as pa


def new_record_batch(values):
    sequence = pa.array(values)
    record_batch = pa.RecordBatch.from_arrays([sequence, sequence], ['a', 'b'])

    return record_batch