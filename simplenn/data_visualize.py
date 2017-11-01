import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import pdb

DEBUG = False
data = pd.read_csv('binary.csv')
if DEBUG:
	pdb.set_trace()
rank1_data = data[data['rank'] == 1]
rank2_data = data[data['rank'] == 2]
rank3_data = data[data['rank'] == 3]
rank4_data = data[data['rank'] == 4]
if DEBUG:
	pdb.set_trace()

color_green = 'green'
color_red = 'red'

fig, axes = plt.subplots(nrows=2, ncols=2)
count = 0
axes = axes.flatten()
for rank_data in [rank1_data, rank2_data, rank3_data, rank4_data]:
	axes[count].scatter(rank_data[rank_data['admit'] == 1]['gpa'], rank_data[rank_data['admit'] == 1]['gre'], c=color_green, label='admitted', alpha=0.7)
	axes[count].scatter(rank_data[rank_data['admit'] == 0]['gpa'], rank_data[rank_data['admit'] == 0]['gre'], c=color_red, label='rejected', alpha=0.7)
	axes[count].legend(loc=4)
	axes[count].set_title('rank' + str(count+1))
	count += 1

fig.tight_layout()
plt.show()

