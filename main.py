from src.main import dock 
from argparse import Namespace
from model.translator import MSLGTranslator
from generate.generation import generate_spa,generate_mslg
from generate.filter import filter_synthetic
import pandas as pd 
from generate.refine import refine_with_bart
from utils.IO import save_json,merge_datasets
def main(args):
    print("[INFO] inicio entrenamiento inicial")
   # dock(args)
    args.flag=False
    print("[INFO] inicio cargando modelo inicial")
    model=MSLGTranslator(args.output_dir)
    df=pd.read_csv("data/processed.csv")
    print("[COMPUTE]generando español")
    synthetic_spa=generate_spa(df,model)
    print("[COMPUTE]generando mslg")
    synthetic_msgl=generate_mslg(df,model)
    synthetic=synthetic_msgl+synthetic_spa
    print("[COMPUTE] refinando con BART ")
    refined = refine_with_bart(synthetic)
    print("[COMPUTE]Se filtra")
    filtered=filter_synthetic(refined)
    save_json(filtered,"data/synthetic.json")
    combined_path="data/combined.txt"
    print("[COMPUTE] Se une los datasets")
    merge_datasets(args.src_input,"data/synthetic.json",combined_path)
    args.src_input=combined_path
    print("[COMPUTE] Se entrena de nuevo")
    dock(args)

if __name__ == "__main__":
    args=Namespace(src_input="data/MSLG_SPA_train.txt",
                   model_name="t5-small",
                   output_dir=f"outputs/model_v0",
                   flag=True,
                   output_dir_1=f"outputs/model_v1")
    main(args)
