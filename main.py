from src.main import dock 
from argparse import Namespace
def main(args):
    dock(args)


if __name__ == "__main__":
    args=Namespace(src_input="data/MSLG_SPA_train.txt",
                   model_name="t5-small",
                   output_dir=f"outputs/model_v0")
    main(args)
