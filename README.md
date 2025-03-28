# Flow-Match-3d-Gaussian: Transforming a Gaussian into a Multi-Modal Gaussian with Flow Matching

![simulation](https://github.com/user-attachments/assets/a594c240-067b-400d-a9c7-1f27c9cbec5e)

This repository implements **flow matching** to transform a 3D Gaussian distribution into a multi-modal Gaussian distribution using a neural network to approximate the marginal vector field. The transformation is achieved by simulating particle flow via the Euler method, and the results are visualized in 3D to show the evolution of the distribution over time.

## Project Overview

Flow matching is a generative modeling technique that learns a vector field to transport samples from a source distribution to a target distribution. In this project:
- **Source Distribution**: A 3D standard Gaussian \( \mathcal{N}(0, I) \).
- **Target Distribution**: A multi-modal Gaussian distribution in 3D space.
- **Method**: A neural network (MLP) parameterizes the vector field, which is trained to minimize the difference between the predicted and true velocities along the flow path.
- **Simulation**: The learned vector field is used to simulate particle trajectories from \( t=0 \) to \( t=1 \) using the Euler method.
- **Visualization**: 3D scatter plots illustrate the transformation of the distribution.

This project demonstrates the power of flow matching for distribution transformation and provides a clear, visual way to understand the process.

## Features
- Implementation of flow matching with a neural network in PyTorch.
- Transformation of a single-mode Gaussian into a multi-modal Gaussian.
- 3D visualization of the particle flow at different time steps.
- Training script with loss monitoring and visualization.

### Training Loss
The training loss curve shows the convergence of the neural network as it learns the vector field:

![loss_curve](https://github.com/user-attachments/assets/eba5c125-fb7e-4c94-9964-35e3cbda7dd8)

## Installation

### Prerequisites
- Python 3.11 or higher
- PyTorch
- Matplotlib
- NumPy
- tqdm (for progress bars)

### Setup
Clone the repository:
   ```bash
   git clone https://github.com/inventwithdean/flow-match-3d-gaussian.git
   cd flow-match-gaussian
   ```

## Usage

### 1. Train the Model
Run the training script to learn the vector field:
```bash
python train.py
```
This will:
- Sample points from the target multi-modal Gaussian distribution.
- Train the neural network to approximate the vector field.
- Save the trained model's state dict as `vf`.
- Generate a loss curve plot (`loss_curve.png`).

### 2. Generate Visualizations
Run the visualization script to simulate and visualize the particle flow:
```bash
python simulate.py
```
This will:
- Load the trained model.
- Simulate the flow of 3,000 particles from \( t=0 \) to \( t=1 \).
- Save the simulation video as `out.mp4`.

## Implementation Details

- **Dataset**: The target distribution is a multi-modal Gaussian, created by sampling from a mixture of Gaussians in 3D space.
- **Model**: A multi-layer perceptron (MLP) with ReLU activations, taking 4 inputs (\( x, y, z, t \)) and outputting a 3D velocity vector.
- **Training**: The vector field is trained using flow matching, minimizing the MSE between the predicted and true velocities along straight-line paths.
- **Simulation**: The Euler method is used to integrate the ODE \( \frac{dx}{dt} = v(x, t) \) from \( t=0 \) to \( t=1 \).
- **Visualization**: Matplotlib is used to create 3D scatter plots of particle positions at various time steps.

## Future Work
- Experiment with more complex target distributions (e.g., 3D shapes like a torus).
- Implement advanced ODE solvers (e.g., using `torchdiffeq`) for more accurate simulations.
- Explore conditional flow matching for controlled transformations.

## Contributing
Contributions are welcome! If you have ideas for improvements or new features, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.

## Acknowledgments
- Inspired by recent work on flow matching for generative modeling.
- Thanks to the PyTorch and Matplotlib communities for their excellent tools.

## Contact
Feel free to reach out via GitHub issues or connect with me on [LinkedIn](https://www.linkedin.com/in/inventwithdean) for questions or collaboration opportunities!
