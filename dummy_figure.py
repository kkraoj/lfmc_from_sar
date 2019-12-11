# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 14:41:48 2019

@author: kkrao
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter


mediumname = 'presentation'
plt.style.use(mediumname)
# Create dummy data
x = np.linspace(0, 14, 100)
fig, ax = plt.subplots()
for i in range(1, 4):
    ax.plot(x, np.sin(x + i * .5) * (7 - i),\
                label = i, markevery = 15,marker = 'o')
ax.legend(loc = 'lower right')
ax.set_xlim(0,19)
ax.set_title('Plot for %s'%mediumname)
ax.set_xlabel('Time (s)')
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
ax.set_ylabel('Amplitude (dB)')
    