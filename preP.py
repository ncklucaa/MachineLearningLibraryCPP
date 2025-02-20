import pandas as pd
import numpy as np

df = pd.read_csv("dataIn.csv")

df = (df - df.min()) / (df.max() - df.min())

df.to_csv("normalized_data.csv", index=False)

print("saved as normalized_data.csv")