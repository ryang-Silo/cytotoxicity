import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import argparse

WORKDIR = os.getcwd()

def generate_compound_images(args):
    data_path = args.src
    
    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=WORKDIR+'/data', type=str)
    
    
    args = parser.parse_args()
    generate_compound_images(args)