## Introduction to PyTorch
---

### **Unit 1: Introduction to PyTorch**
- **What is PyTorch?**
  - Open-source machine learning library for Python.
  - Widely used for deep learning and neural network tasks.
- **Key Features:**
  - Dynamic computation graph (eager execution).
  - GPU acceleration (leverages CUDA), check using `torch.cuda.is_available()`
  - Popular for research and development.

---

### **Unit 2: Tensors**
- **Tensors** are the core data structure in PyTorch, similar to NumPy arrays but with GPU support.
  - 0D: Scalar.
  - 1D: Vector.
  - 2D: Matrix.
  - nD: General n-dimensional array.
- **Creating Tensors:**
  - `torch.tensor([1, 2, 3])` → creates a 1D tensor.
  - Methods like `torch.zeros()`, `torch.ones()`, `torch.rand()`.
- **Operations on Tensors:**
  - Element-wise operations: `+`, `-`, `*`, `/`.
  - Matrix operations: `torch.matmul()` for matrix multiplication.
  - Supports broadcasting like NumPy.

---

### **Unit 3: Autograd**
- **Autograd** is PyTorch’s automatic differentiation system.
  - Allows automatic computation of gradients (for backpropagation).
  - **Tracking gradients:** `requires_grad=True` on tensors.
  - **Backward pass:** `.backward()` method calculates gradients.
- **Gradient Calculation:**
  - `grad` attributes store gradients.
  - Essential for optimization algorithms like gradient descent.

---

### **Unit 4: Building Neural Networks**
- PyTorch provides **torch.nn** module for building neural networks.
- **Model Structure:**
  - Defined by subclassing `torch.nn.Module`.
  - Layers are defined in the `__init__` method.
  - Forward pass is defined in the `forward()` method.
- Example Layers:
  - `torch.nn.Linear`: Fully connected (dense) layer.
  - `torch.nn.ReLU`: Activation function.
  - `torch.nn.Conv2d`: Convolutional layer for image data.

---

### **Unit 5: Loss Functions and Optimizers**
- **Loss Functions:** Measure the difference between the model's predictions and the actual values.
  - `torch.nn.MSELoss()` for regression.
  - `torch.nn.CrossEntropyLoss()` for classification.
- **Optimizers:** Used to update the model weights to minimize the loss.
  - `torch.optim.SGD`: Stochastic Gradient Descent.
  - `torch.optim.Adam`: Adaptive learning rate optimizer.
- **Training Loop:**
  1. Forward pass to compute predictions.
  2. Calculate loss.
  3. Backward pass to compute gradients.
  4. Update weights using optimizer.

---

### **Unit 6: Training a Neural Network**
- **Steps for training a network:**
  1. **Prepare data**: Use `torch.utils.data.DataLoader` for batching.
  2. **Build the model**: Define layers and architecture.
  3. **Define the loss function**: Choose appropriate loss (e.g., Cross Entropy for classification).
  4. **Choose optimizer**: Like Adam, SGD, etc.
  5. **Training Loop**:
     - For each epoch:
       - Pass input through the network (forward pass).
       - Compute loss.
       - Perform backpropagation (`loss.backward()`).
       - Update weights (`optimizer.step()`).
  6. **Evaluate performance**: Validate and test the model on unseen data.

---

