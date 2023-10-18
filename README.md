# Gaussian Optimization on a 2D Image

Reconstructs a grayscale image using a combination of 2D Gaussians. Initializes a set of Gaussians over the image, then refines their parameters (e.g., position, spread, orientation, color) to best match the original image.

## Result
<img src="https://raw.githubusercontent.com/uynitsuj/2DGaussianOptimization/main/data/flwrrecon4.gif" width="500" height="500">

Original Image:

<img src="https://raw.githubusercontent.com/uynitsuj/2DGaussianOptimization/main/data/output31.png" width="400" height="400">

Gaussians after 1 iteration:

<img src="https://raw.githubusercontent.com/uynitsuj/2DGaussianOptimization/main/data/output32.png" width="400" height="400">

Gaussians after 300 iterations:

<img src="https://raw.githubusercontent.com/uynitsuj/2DGaussianOptimization/main/data/output33.png" width="400" height="400">

## Libraries
```python
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio
from IPython.display import Video, display
```

## 2D Gaussian Function
Defines a 2D Gaussian function with parameters for position, spread, intensity, and rotation.

```python
def gaussian_2d(x, y, mu_x, mu_y, sigma_x, sigma_y, color_intensity, rotation_angle):
```

## Loss Function
Defines a Mean Squared Error (MSE) loss function to measure the difference between the original image and the Gaussian-based reconstruction.

```python
def mse_loss(params):
```

## Optimization
Uses the Adam optimizer to refine the Gaussians' parameters for better image reconstruction. Also employs a learning rate scheduler for better convergence.

```python
optimizer = optim.Adam([params], lr=0.5) 
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.7)
```
