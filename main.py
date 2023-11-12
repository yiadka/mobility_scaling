import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from datetime import timedelta
import pickle

from src import preprocessing as pre

def main():
    print("+------------------+")
    print("| Loading datasets |")
    print("+------------------+")
    df_left, df_right = pre.preprocess('data/df.pickle')

    # それぞれ1000件ずつ抽出
    df_left_sample = df_left.sample(n=1000, random_state=0)
    df_right_sample = df_right.sample(n=1000, random_state=0)
    
    print("+------------------+")
    print("| Done             |")
    print("+------------------+")
    print(df_left.head())

    print("+------------------+")
    print("| FINISH         |")
    print("+------------------+")

if __name__ == '__main__':
    main()
    