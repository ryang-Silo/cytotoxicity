import os
import sys
import numpy as np
import cv2
import json
from pathlib import Path
import argparse


WORKDIR = os.getcwd()
print(f"working dir: {WORKDIR}")


def target_function(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataPath", default=os.path.join(WORKDIR, 'eda_phase/lbp_experiments/data/test'), type=str)
    parser.add_argument("--outPath", default=WORKDIR+'/eda_phase/lbp_experiments/test_features', type=str)
    
    args = parser.parse_args()
    
    target_function(args)