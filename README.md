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

