def table_to_dict_of_numpy(table):
    table_pandas = table.to_pandas()

    return {column: table_pandas[column].values for column in table_pandas.columns}
