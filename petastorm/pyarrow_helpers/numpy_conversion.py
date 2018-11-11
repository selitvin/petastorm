import numpy as np
import pyarrow as pa

def batch_table_to_dict_of_numpy(table, unischema):

    numpy_batches = {}
    for column in table.columns:
        unischema_field = unischema.fields[column.name]
        numpy_batches[column.name] = unischema_field.codec.arrow_column_to_numpy_batch(unischema_field, column.data)

    return numpy_batches

    #
    # table_pandas = table.to_pandas()
    #
    # return {column: table_pandas[column].values for column in table_pandas.columns}

def table_to_dict_of_numpy(table, unischema):
    reshaped_row = {}
    for column in table.columns:
        unischema_field = unischema.fields[column.name]

        data = column.data[0]
        if data is pa.NULL:
            cell = None
        else:
            # TODO(yevgeni): this means we don't support shapes with more than one unknown dimension (None, None)
            cell = unischema_field.codec.arrow_value_to_numpy(unischema_field, data)

        reshaped_row[column.name] = cell

    return reshaped_row

_NUMPY_TO_PYARROW_TYPES = {
    np.uint8: pa.uint8(),
    np.int8: pa.int8(),
    np.uint16: pa.uint16(),
    np.int16: pa.int16(),
    np.uint32: pa.uint32(),
    np.int32: pa.int32(),
    np.uint64: pa.uint64(),
    np.int64: pa.int64(),
    np.float16: pa.float16(),
    np.float32: pa.float32(),
    np.float64: pa.float64(),
    np.bool_: pa.bool_(),
    np.bytes_: pa.string(),
    np.str_: pa.string(),
    np.unicode_: pa.string(), # TODO(yevgeni): unicode vs bytes
}
def get_pyarrow_type_from_numpy(numpy_type):
    return _NUMPY_TO_PYARROW_TYPES[numpy_type]
