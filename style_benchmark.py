import numpy as np
import polars as pl
from benchmark import load_data
from original_style import construct_style_matrices
from faster_style import construct_style_matrices as construct_style_matrices_fast

def main():
    # df = load_data(use_polars=True, N=10000)
    df = pl.read_parquet("processed_data.parquet").to_pandas()
    # print(df.head(1)['conv_metadata'].to_dict())

    # X, y, models = construct_style_matrices(df)
    # print(X.shape, y.shape)
    # print(models)

    construct_style_matrices_fast(df)

if __name__ == '__main__':
    main()