import pandas as pd
from original import get_bootstrap_result as og_bootstrap
from faster import get_bootstrap_result as fast_bootstrap

def main():
    # lol
    df = pd.read_json('local_file_name.json', nrows=1000).sort_values(ascending=True, by=["tstamp"])
    print(len(df))

if __name__ == '__main__':
    main()