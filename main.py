from src.main import dock 
from argparse import Namespace
def main(args):
    dock(args)


if __name__ == "__main__":
    args=Namespace(src_input="data/MSLG_SPA_train.txt")
    main(args)
