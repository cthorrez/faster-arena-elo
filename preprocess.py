import polars as pl

def main():
    df = pl.read_json('clean_battle_20240826_public.json').lazy().filter(pl.col("anony"))
    df = df.filter(pl.col("dedup_tag").struct.field("sampled").fill_null(False))
    df = df.sort("tstamp")
    df = df.select("model_a", "model_b", "winner", "conv_metadata")
    df = df.collect()
    df.write_parquet('processed_data.parquet')

if __name__ == '__main__':
    main()