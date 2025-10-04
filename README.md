# 🧮 Computational Graph with PyTorch (Manual Forward & Backward)

## 📌 Overview
This project demonstrates how to **manually build a computational graph** using **PyTorch tensors** with `requires_grad=True`.  
Instead of using `nn.Module` or predefined layers, we explicitly define **weights, biases, and activations** to understand how the forward and backward passes work in deep learning.

The network follows this structure:

```
Input (x)
│
├── Layer 1: 3 neurons (w00,b00 / w01,b01 / w02,b02)
│     ↓ Activation: ReLU
│
├── Layer 2: 2 neurons (w10,b10 / w11,b11)
│     ↓ Activation: Sigmoid
│
├── Combine the two outputs (+)
│     ↓ Activation: Tanh
│
└── Output Layer: 1 neuron (w20, b20)
      ↓ Activation: None (linear output)
```

---

## ⚙️ Requirements
- Python 3.x  
- [PyTorch](https://pytorch.org/)  

Install dependencies:
```bash
pip install torch
```

---

## 🏗️ Step-by-step Flow
1. **Define tensors for input, weights, and biases**  
   Each parameter is a `torch.tensor` with `requires_grad=True`.

2. **Layer 1 (3 neurons + ReLU)**  
   - Compute `z = w*x + b` for each neuron  
   - Apply `torch.relu`  

3. **Layer 2 (2 neurons + Sigmoid)**  
   - Compute outputs from Layer 1  
   - Apply `torch.sigmoid`

4. **Combine outputs (+) → Apply Tanh**  
   - Sum outputs from Layer 2  
   - Apply `torch.tanh`

5. **Output layer (linear)**  
   - Compute final linear combination `w * h + b`

6. **Backward pass**  
   - Call `.backward()` on the final output  
   - Print gradient of the output w.r.t input  

---

## 📜 Example Code
```python
import torch

# Input
x = torch.tensor([1.0], requires_grad=True)

# Layer 1 weights & biases
w00, b00 = torch.tensor(0.5, requires_grad=True), torch.tensor(0.1, requires_grad=True)
w01, b01 = torch.tensor(-0.3, requires_grad=True), torch.tensor(0.2, requires_grad=True)
w02, b02 = torch.tensor(0.8, requires_grad=True), torch.tensor(-0.5, requires_grad=True)

# Layer 2 weights & biases
w10, b10 = torch.tensor(0.7, requires_grad=True), torch.tensor(0.05, requires_grad=True)
w11, b11 = torch.tensor(-0.6, requires_grad=True), torch.tensor(-0.1, requires_grad=True)

# Output layer weights & bias
w20, b20 = torch.tensor(1.2, requires_grad=True), torch.tensor(0.3, requires_grad=True)

# --- Forward Pass ---

# Layer 1
h1_0 = torch.relu(w00 * x + b00)
h1_1 = torch.relu(w01 * x + b01)
h1_2 = torch.relu(w02 * x + b02)

print("Layer 1 outputs:", h1_0.item(), h1_1.item(), h1_2.item())

# Layer 2
h2_0 = torch.sigmoid(w10 * h1_0 + b10)
h2_1 = torch.sigmoid(w11 * h1_1 + b11)

print("Layer 2 outputs:", h2_0.item(), h2_1.item())

# Combine + Tanh
combined = torch.tanh(h2_0 + h2_1)
print("Combined output:", combined.item())

# Output layer (linear)
output = w20 * combined + b20
print("Final output:", output.item())

# --- Backward Pass ---
output.backward()

print("Gradient of output w.r.t input x:", x.grad.item())
```

---

## 📊 Expected Behavior
- Prints intermediate outputs after each layer.  
- Shows the **final scalar output**.  
- Computes and prints the **gradient of the output w.r.t the input**.  

---

## 🎯 Learning Goals
- Understand how **manual computational graphs** are built in PyTorch.  
- Learn how to use **activation functions (ReLU, Sigmoid, Tanh)** step by step.  
- Perform **backpropagation manually** using `.backward()`.  
