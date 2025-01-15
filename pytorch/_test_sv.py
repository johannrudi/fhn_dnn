import numpy as np
import matplotlib.pyplot as plt

out_features  = np.load('singular_values.npy')
out_vectors  = np.load('singular_vectors.npy')
print(out_vectors.shape)

fig, ax = plt.subplots(4, 1)
for i in range(4):
    ax[i].plot(out_vectors[:,50+i])
# plt.plot(out_vectors)
# plt.yscale("log")
# plt.xlim([0, 200])
# plt.ylim([1e-1, 1e3])

plt.show()