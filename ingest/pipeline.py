from ingest.read import read
from ingest.preprocessing import pipeline_mslg
def pipeline(src):
    df=read(src)
    df["MSLG"]=df["MSLG"].apply(pipeline_mslg)
    print(df["MSLG"])
    df.to_csv("data/cleaned.csv",index=False)