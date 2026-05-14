

# plot sample swarmplto

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("sample.csv")

plt.figure(figsize=(10, 5))
sns.swarmplot(x="condition", y="value", data=df)
plt.show()
