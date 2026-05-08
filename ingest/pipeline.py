from ingest.read import read
from ingest.normalization import normalize_by_task
from ingest.preprocessing import pipeline_mslg
from ingest.dataset import assign_splits,build_multitask,df_to_hf_datasets
import pandas as pd 
from augmentation.pipeline import augment_df
def pipeline(src):
    print("[INFO] inicio preprocesamiento ligero")
    df=read(src)
    df["MSLG"]=df["MSLG"].apply(pipeline_mslg)
    print("[INFO] inicio tareas")
    df=assign_splits(df)
    df=build_multitask(df)
    print("[INFO] inicio el aumento de datos con fasttext")
    df=augment_df(df,ft_model_path="cc.es.300.bin")
    print("[INFO] inicio la normalziacion")
    df=df.apply(normalize_by_task,axis=1)
    df.to_csv("data/processed.csv", index=False)
    return df_to_hf_datasets(df)
