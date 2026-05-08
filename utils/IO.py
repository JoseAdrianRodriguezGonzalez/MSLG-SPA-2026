import json
import pandas as pd 
from ingest.read import read
def save_json(data,path):
    with open(path,"w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2)
def read_json(path):
    with open(path,"r",encoding="utf-8") as f:
        json_result=json.load(f)
    return json_result
def merge_datasets(original_path,synthetic_path,output_path):
    df_original=read(original_path)
    json_synthetic=read_json(synthetic_path)
    df_synth=pd.DataFrame(json_synthetic)
    df_synth=df_synth[["SPA","MSLG"]]
    df_combined=pd.concat([df_original,df_synth],ignore_index=True)
    df_combined.to_csv(output_path, sep="\t", index=False)
    return output_path
