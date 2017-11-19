import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3.0, 3.0, 1000)

y1 = np.sin(x)
ep = np.exp(x)
en = np.exp(-x)
y2 = 1. / (1. + en)
y3 = (ep - en) / (ep + en)

plt.plot(x, np.zeros_like(x), 'k-', label='x')
plt.plot(np.zeros_like(x), x, 'k-')
plt.plot(x, x, 'r-')
plt.plot(x, y1, 'g-')
plt.plot(x, (y2 - 0.5)*4, 'b-')
plt.plot(x, y3, 'm-')

plt.show()
