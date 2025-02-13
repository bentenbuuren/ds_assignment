from argparse import ArgumentParser
from joblib import load
import os

def test(test1,test2):
    print(test1,test2)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("training_dataset", type=str)
    parser.add_argument("model_filename", type=str)
    args = parser.parse_args()
    test(args.training_dataset, args.model_filename)
