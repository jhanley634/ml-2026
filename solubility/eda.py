#! /usr/bin/env YDATA_PROFILING_BANNER=0 python

# from:
# https://www.kaggle.com/code/kerneler/starter-aqsoldb-a-curated-aqueous-6146b12d-1

import os
from pathlib import Path

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from ydata_profiling import ProfileReport

dataset = "sorkun/aqsoldb-a-curated-aqueous-solubility-dataset"
folder = Path(kagglehub.dataset_download(dataset))
csv = folder / "curated-solubility-dataset.csv"

df = pd.read_csv(csv)
print(df)
print(df.describe())

profile = ProfileReport(df, title="YData Profiling Report")
html = Path("/tmp/solubility.html")
if not html.exists():
    profile.to_file(html)
