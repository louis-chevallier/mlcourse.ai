import numpy as np
import pandas as pd
import os
from utillc import *
pd.set_option("display.precision", 2)
file = os.path.join("traffic_accidents.csv")
file = "/home/louis/Desktop/traffic_accidents.csv"
df = pd.read_csv(file)
EKOX(df.head())
