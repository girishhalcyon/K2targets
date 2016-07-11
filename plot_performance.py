import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

labels = ['Kp', 'J', 'H', 'K', r'$\mu$', '$b$', 'u', 'g', 'r', 'i', 'z']

column_labels = labels
row_labels = labels

data = np.array([[0.0,0.9056,0.9302,0.9305, 0.5039, 0.71, 0.3511, 0.9176, 0.5284, 0.8956, 0.8570],
    [0.9506,0.0,0.8585,0.8742, 0.5121, 0.7411, 0.3689, 0.9411, 0.9114, 0.87, 0.8],
    [0.9302,0.8585,0.0,0.5946, 0.5314, 0.7533, 0.4212, 0.9512, 0.9347, 0.9164, 0.8909],
    [0.9305,0.8742,0.5946,0.0, 0.5358, 0.7749, 0.4296, 0.9526, 0.9366, 0.9174, 0.8944],
    [0.5039, 0.5121, 0.5314, 0.5358, 0.0, 0.3986, 0.263, 0.5491, 0.5192, 0.522, 0.4837],
    [0.71, 0.7411, 0.7533, 0.7749, 0.3986, 0.0, 0.2576, 0.7658, 0.7208, 0.7389, 0.6863],
    [0.3511, 0.3689, 0.4212, 0.4296, 0.2630, 0.2576, 0.0, 0.3905, 0.3755, 0.3593, 0.3497],
    [0.9176, 0.9411, 0.9512, 0.9526, 0.5491, 0.7658, 0.3905, 0.0, 0.9224, 0.9347, 0.9197],
    [0.5284, 0.9114, 0.9347, 0.9366, 0.5192, 0.7208, 0.3755, 0.9224, 0.0, 0.8828, 0.86],
    [0.8956, 0.87, 0.9164, 0.9174, 0.522, 0.7389, 0.3593, 0.9347, 0.8828, 0.0, 0.7279],
    [0.857, 0.8, 0.8909, 0.8944, 0.4837, 0.6863, 0.3497, 0.9197, 0.86, 0.7279, 0.0]])

test = np.empty((11, 11))
for i in range(0,11):
    test[i,:] = data[i]
print test.shape

fig, ax = plt.subplots()
heatmap = ax.pcolor(data, cmap='Greys')
fig.colorbar(heatmap)
# put the major ticks at the middle of each cell

# want a more natural, table-like display
ax.invert_yaxis()
ax.xaxis.tick_top()

ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)

ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(column_labels, minor=False)

ax.yaxis.grid(True, which='minor', linestyle='-', color='k', alpha = 0.5)
ax.xaxis.grid(True, which='minor', linestyle='-', color='k', alpha = 0.5)

# Set the location of the minor ticks to the edge of pixels for the x grid
minor_locator = AutoMinorLocator(2)
ax.xaxis.set_minor_locator(minor_locator)

# Lets turn off the actual minor tick marks though
for tickmark in ax.xaxis.get_minor_ticks():
    tickmark.tick1On = tickmark.tick2On = False

# Set the location of the minor ticks to the edge of pixels for the y grid
minor_locator = AutoMinorLocator(2)
ax.yaxis.set_minor_locator(minor_locator)

# Lets turn off the actual minor tick marks though
for tickmark in ax.yaxis.get_minor_ticks():
    tickmark.tick1On = tickmark.tick2On = False
plt.axis('tight')
plt.suptitle('Accuracy of RandomForestClassifier with Kepler Sample (Trained on 60%, Tested on 20%)')
plt.savefig('RandomForestScore.pdf', dpi = 3000)
