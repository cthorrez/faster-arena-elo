import math
import numpy as np
import polars as pl
from benchmark import load_data
from original_style import construct_style_matrices, fit_mle_elo
from faster_style import construct_style_matrices as construct_style_matrices_fast, fit_contextual_bt

def main():
    N = 20000
    # df = load_data(use_polars=True, N=N)
    df = pl.scan_parquet("processed_data.parquet", n_rows=N).collect().to_pandas()
    # print(df.head(1)['conv_metadata'].to_dict())

    X, y, models = construct_style_matrices(df)
    output = fit_mle_elo(X, y, models)
    print(output)


    matchups, features, outcomes, models = construct_style_matrices_fast(df)
    ratings, feature_params = fit_contextual_bt(
        matchups,
        features,
        outcomes,
        weights=np.ones_like(outcomes),
        models=models,
        alpha=math.log(10.0),
        reg=1.0,
        init_rating=1000.0,
        scale=400.0,
    )
    for rank_idx in np.argsort(-ratings):
        print(models[rank_idx], ratings[rank_idx])
    
    print(feature_params)



if __name__ == '__main__':
    main()