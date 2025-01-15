
# Perceptron Training Script

This repository contains a Python script for training a perceptron model using a supervised learning approach.

## Features

- Implements the Perceptron learning algorithm.
- Configurable learning rate and number of iterations.
- Supports binary classification tasks.

## Requirements

To run this script, you need the following installed on your system:

- Python 3.x
- NumPy (for numerical computations)
- Matplotlib (for optional visualization)

## How to Use

1. **Clone the repository:**

   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. **Install dependencies:**

   Install the required Python libraries using pip:

   ```bash
   pip install numpy matplotlib
   ```

3. **Run the script:**

   Execute the script with:

   ```bash
   python Perceptron_training.py
   ```

4. **Customize the Parameters:**

   Edit the script to adjust the following parameters as needed:
   - Learning rate
   - Number of iterations
   - Input data file or dataset generation parameters

## Script Overview

### Functions

- **`initialize_weights()`**: Initializes the weights and bias for the perceptron model.
- **`train_perceptron()`**: Trains the perceptron using the specified dataset.
- **`predict()`**: Makes predictions on new data points.

### Input Data

The script expects the input data to be in the following format:
- **Features**: A 2D array where each row is a data sample.
- **Labels**: A 1D array containing binary labels (e.g., 0 or 1).

### Output

- **Final weights and bias**: Displayed in the console.
- **Training progress**: Optionally plotted using Matplotlib.

## Example

Here is an example of running the perceptron training:

```python
# Sample dataset
features = [[0, 0], [0, 1], [1, 0], [1, 1]]
labels = [0, 0, 0, 1]

# Training the perceptron
train_perceptron(features, labels, learning_rate=0.1, iterations=10)
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

Inspired by the classic perceptron algorithm used in machine learning and artificial intelligence.

## Contact

For issues or questions, please contact [Your Name] at [your_email@example.com].
