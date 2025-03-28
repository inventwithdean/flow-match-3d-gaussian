import torch
from torch.distributions.multivariate_normal import  MultivariateNormal
from vector_field import VectorField
import matplotlib.pyplot as plt
from matplotlib.animation import  FuncAnimation

field = VectorField()
field.load_state_dict(torch.load("./vf"))
field.eval()

# Standard Normal Distribution
dist = MultivariateNormal(torch.zeros(3), torch.eye(3))


fig, ax = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
ax[0].view_init(elev=30, azim=60)
ax[0].set_title("Initial Distribution")
ax[1].view_init(elev=30, azim=60)
ax[1].set_title("Distribution after solving ODE")
inputs = dist.sample((1000,))
plot_inputs = inputs.detach().numpy()
ax[0].scatter(inputs[:, 0], inputs[:, 1], inputs[:, 2], alpha=0.3)

# Simulating ODE
steps = 100
h = 1/steps
for i in range(steps):
    t = i*h
    batch_size = inputs.shape[0]
    time_points = torch.ones((batch_size, 1))*t
    vectors = field(inputs, time_points)
    inputs = inputs + h*vectors

plot_inputs = inputs.detach().numpy()
print("PLOTTING...")
print(plot_inputs.shape)
ax[1].scatter(plot_inputs[:, 0], plot_inputs[:, 1], plot_inputs[:, 2], alpha=0.3)
fig.savefig("./simulation.png", dpi=300)

# ANIMATION

fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
ax.set_title("Flow Matching: Transforming a Gaussian into a Multi-Modal Gaussian\n via a Neural Vector Field")

ax.view_init(elev=30, azim=0)

ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-5, 5)

inputs = dist.sample((3000,))
plot_inputs = inputs.numpy()
plot = ax.scatter(plot_inputs[:, 0], plot_inputs[:, 1], plot_inputs[:, 2], alpha=0.3)
ax.scatter([3.0], [3.0], [-3.0], marker="x", s=36, color="red")
ax.scatter([-3.0], [0.0], [-1.5], marker="x", s=36, color="red")
ax.scatter([-2.0], [-2.0], [2.0], marker="x", s=36, color="red")

def init():
    plot._offsets3d = (plot_inputs[:, 0], plot_inputs[:, 1], plot_inputs[:, 2])

completion_frames = 360
total_frames = completion_frames + 360

def update(frame_num, positions):
    t = (frame_num/completion_frames)
    ax.view_init(azim=t*60)
    if t > 1:
        return None
    batch_size = positions.shape[0]
    time_points = torch.ones((batch_size, 1)) * t
    vectors = field(positions, time_points)
    new_positions = positions + (1/completion_frames)*vectors
    positions[:] = new_positions
    numpy_new_positions = new_positions.detach().numpy()
    plot._offsets3d = (numpy_new_positions[:, 0], numpy_new_positions[:, 1], numpy_new_positions[:, 2])
    return new_positions

anim = FuncAnimation(fig, update, total_frames, init_func=init, fargs=(inputs,), interval=1)

print("SAVING!")
anim.save("./out.mp4", fps=30, dpi=300)