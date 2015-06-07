import pandas as pd
import matplotlib.pyplot as plt
import pylab as P

df = pd.read_csv('K:\python_workspace\\titanic\\train.csv', header=0)
df['Age'].dropna().hist(bins=16, range=(0,80), alpha = .5)
plt.show()

