import pandas as pd

df = pd.read_fwf('/data/poohbah/1/assassin/lenhart/code/calder/vsxcat.090525')
df.to_csv('/data/poohbah/1/assassin/lenhart/code/calder/vsxcat_csv.090525', index=False)