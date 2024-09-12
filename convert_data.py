import pandas as pd

def main():
    df = pd.read_json('local_file_name.json').sort_values(ascending=True, by=["tstamp"])
    df = df[df["anony"] == True]
    df = df[df["dedup_tag"].apply(lambda x: x.get("sampled", False))]
    df.to_parquet('data.parquet', index=False)

if __name__ == '__main__':
    main()