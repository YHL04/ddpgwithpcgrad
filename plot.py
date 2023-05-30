import pandas as pd
import os
import matplotlib.pyplot as plt

dir = os.listdir("logs/")
dir.sort()
latestfile = "logs/" + dir[-1]

data = pd.read_csv(latestfile,
                   names=["Reward"]
                   )

plt.subplot(1, 1, 1)
plt.plot(data["Reward"])

plt.show()
