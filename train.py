import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal

from dataset import GaussianDataset
from vector_field import VectorField
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

training_data = GaussianDataset()
train_dataloader = DataLoader(training_data, batch_size=512, shuffle=True)

num_epochs = 7

# Source Distribution: 3D Standard Gaussian N(0, Id)
# Standard normal Distribution to generate epsilon
dist = MultivariateNormal(loc=torch.zeros(3), covariance_matrix=torch.eye(3))

vectorField = VectorField()

mse = nn.MSELoss()

optimizer = torch.optim.Adam(vectorField.parameters(), lr=1e-3)
losses = []
for epoch in range(num_epochs):
    step = 0
    for targets in tqdm(train_dataloader):
        step += 1
        optimizer.zero_grad()
        batch_size = targets.shape[0]
        t = torch.rand((batch_size, 1))

        # Training on Conditional Vector Field
        # Gaussian Distribution conditional towards current target point
        # At time point t, probability path density p_t(x|z) will be N(t*alpha, (1-t)**2 * beta)
        # So we need to sample a point from here and calculate the true vector pointing towards target
        # Can be t*z + (1-t)*epsilon, where
        # epsilon ~ N(0, 1) by reparameterization trick
        epsilon = dist.sample((batch_size,))

        # Vector field will be (z - epsilon)
        initial_points = t * targets + (1-t) * epsilon
        outputs = vectorField(initial_points, t)
        target_vectors = (targets - epsilon)

        loss = mse(outputs, target_vectors)
        loss.backward()

        optimizer.step()

        if step % 2 == 0:
            losses.append(loss.detach().cpu())

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(losses)
fig.savefig("./loss_curve.png", dpi=300)
torch.save(vectorField.state_dict(), "./vf")