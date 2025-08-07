#! /usr/bin/env python

import kagglehub

path = kagglehub.dataset_download("sorkun/aqsoldb-a-curated-aqueous-solubility-dataset")

print("Path to dataset files:", path)
