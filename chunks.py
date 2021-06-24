import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import sys

path = sys.argv[1]
file = sys.argv[2]
type = sys.argv[3]

if type == "train":
    chunksize = 7000
if type == "val" or type == "test":
    chunksize = 1500

initdataframe = pd.read_csv(path + file, chunksize=chunksize)
# initdataframe.head()
cnt = 0
for chunk in initdataframe:
    print("chunk " + str(cnt))

    if cnt < 10:
        chunk.to_csv(path + "chunks/" + "_chunk_0" + str(cnt) + ".csv")
    else:
        chunk.to_csv(path + "chunks/" + "_chunk_" + str(cnt) + ".csv")
    cnt += 1




