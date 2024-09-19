import os
import io
import polars as pl
import pandas as pd

# to get the data in the first place:
# mkdir data
# cd data
# wget https://storage.googleapis.com/arena_external_data/public/clean_battle_20240826_public.json

DATA_PATH = 'data/clean_battle_20240826_public.json'

def preprocess(write=False):
    df = pl.read_json(DATA_PATH).lazy().filter(pl.col("anony"))
    df = df.filter(pl.col("dedup_tag").struct.field("sampled").fill_null(False))
    df = df.sort("tstamp")
    df = df.select("model_a", "model_b", "winner", "conv_metadata")
    df = df.collect()
    if write:
        # df.write_parquet('data/processed_data.parquet')
        df.write_ndjson('data/processed_data.jsonl')

    else:
        return df


def load_data(N=2_000_000_000, use_polars=False, use_preprocessed=False, route_through_parquet=False):
    if use_preprocessed:
        # df = pl.scan_parquet("data/processed_data.parquet").tail(N).collect().to_pandas()
        df = pl.scan_ndjson("data/processed_data.jsonl").tail(N).collect().to_pandas()

    elif use_polars:
        df = preprocess().tail(N).to_pandas()
    else:
        df = pd.read_json(DATA_PATH).sort_values(ascending=True, by=["tstamp"])
        df = df[df["anony"] == True]
        df = df[df["dedup_tag"].apply(lambda x: x.get("sampled", False))]
        df = df.tail(N)
        df = df.reset_index(drop=True)
        if route_through_parquet:
            buffer = io.BytesIO()
            df.to_parquet(buffer)
            df = pd.read_parquet(buffer)
    print(f"loaded df with: {len(df)} matches")
    return df

if __name__ == '__main__':
    preprocess(write=True)