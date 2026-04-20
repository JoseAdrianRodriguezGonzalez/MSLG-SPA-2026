from ingest.read import read
def pipeline(src):
    df=read(src)
    print(df)