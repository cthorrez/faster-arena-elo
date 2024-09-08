import time
import pandas as pd
from original import get_bootstrap_result as og_bootstrap, compute_mle_elo
from faster import get_bootstrap_result as fast_bootstrap

def main():
    N = 2000
    # lol
    df = pd.read_json('local_file_name.json').sort_values(ascending=True, by=["tstamp"])
    df = df[df["dedup_tag"].apply(lambda x: x.get("sampled", False))]
    df = df.tail(N)
    print(len(df))
    models = pd.unique(df[['model_a', 'model_b']].values.ravel())
    print(models)

    og_results = og_bootstrap(df, compute_mle_elo, 100)
    print(og_results)

if __name__ == '__main__':
    main()