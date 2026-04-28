from sklearn.model_selection import train_test_split
import pandas as pd 
from datasets import Dataset
def assign_splits(df):
    train, temp = train_test_split(df, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"

    return pd.concat([train, val, test], ignore_index=True)
def build_multitask(df):
    df_spa_mslg = pd.DataFrame({
        "input_text": df["SPA"],
        "target_text": df["MSLG"],
        "task": "spa-mslg",
        "split":df["split"]
    })
    df_mslg_spa = pd.DataFrame({
        "input_text": df["MSLG"],
        "target_text": df["SPA"],
        "task": "mslg-spa",
        "split":df["split"]
    })
    return  pd.concat([df_spa_mslg, df_mslg_spa], ignore_index=True)
def df_to_hf_datasets(df):
    train=Dataset.from_pandas(df[df["split"]=="train"])
    val=Dataset.from_pandas(df[df["split"]=="val"])
    test=Dataset.from_pandas(df[df["split"]=="test"])
    return train,val,test