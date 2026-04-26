from ingest.read import read
from ingest.preprocessing import pipeline_mslg
import pandas as pd 
def pipeline(src):
    df=read(src)
    df["MSLG"]=df["MSLG"].apply(pipeline_mslg)
    df1=df.copy()
    df2=df.copy()
    df1["input_text"]=df["SPA"]
    df1["target_text"]=df["MSLG"]
    df1["task"]="spa-mslg"
    
    df2["input_text"]=df["MSLG"] 
    df2["target_text"]=df["SPA"]    
    df1["task"]="mslg-spa"
    df3=pd.concat([df1,df2])
    df3.to_csv("data/processed.csv",index=False)