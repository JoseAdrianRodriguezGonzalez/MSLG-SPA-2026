import pandas as pd 
from model.translator import MSLGTranslator
from tqdm import tqdm 
import os
def read_test(src):
    return pd.read_csv(src,sep="\t")
def generate_predictions(df:pd.DataFrame,model:MSLGTranslator,task:str):
    output=[]
    for _, row in tqdm(df.iterrows(),total=len(df)):
        text=row["SPA"] if task=="spa2mslg" else row["MSLG"]
        pred=model.spa_to_mslg(text) if task=="spa2mslg" else model.mslg_to_spa(text)
        output.append(pred)
    return output 
def save_submission(predictions,output_file):
    with open(output_file,"w",encoding="utf-8") as f:
        for pred in predictions:
            f.write(f"\"{pred}\"\n")
def main_pipeline(args):
    model=MSLGTranslator(args.output_dir_1)
    mslg2spa_df = read_test(args.src_mslg2spa)
    spa2mslg_df = read_test(args.src_spa2mslg)
    print("[INFO] Escribiendo mslg2spa ")
    trad2spa=generate_predictions(mslg2spa_df,model,"mslg2spa")
    print("[INFO] Escribiendo spa2mslg")
    trad2gloss=generate_predictions(spa2mslg_df,model,"spa2mslg")
    assert len(trad2spa) == len(mslg2spa_df)
    assert len(trad2gloss) == len(spa2mslg_df)
    save_submission(trad2gloss,os.path.join(args.entregable,"Lepopardo_run1_SPA2MSLG.txt"))
    save_submission(trad2spa,os.path.join(args.entregable,"Lepopardo_run1_MSLG2SPA.txt"))