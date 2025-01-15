import numpy as np
import matplotlib.pyplot as plt
import torch

out_features  = np.load('out_features.npy')
true_features = np.load('true_features.npy')

idx = 135

plt.plot(out_features[idx].squeeze()) #/np.max(out_features[idx]))
plt.plot(true_features[idx].squeeze()) #/np.max(true_features[idx]))

for i in range(100, 132):
    print("Norm Difference:", np.linalg.norm(out_features[i] - true_features[i]))

print("Norm Difference, torch:", torch.linalg.matrix_norm(
        torch.tensor(out_features[100:132]) -
        torch.tensor(true_features[100:132])
    ).mean().item())

plt.show()