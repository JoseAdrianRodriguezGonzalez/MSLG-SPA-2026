from ingest.read import read
from ingest.preprocessing import pipeline_mslg
from ingest.dataset import assign_splits,build_multitask,df_to_hf_datasets
import pandas as pd 
def pipeline(src):
    df=read(src)
    df["MSLG"]=df["MSLG"].apply(pipeline_mslg)
    df=assign_splits(df)
    df=build_multitask(df)
    df.to_csv("data/processed.csv", index=False)
    return df_to_hf_datasets(df)