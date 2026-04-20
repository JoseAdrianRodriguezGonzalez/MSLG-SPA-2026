import pandas as pd 
def read(src):
    return pd.read_csv(src,sep="\t")