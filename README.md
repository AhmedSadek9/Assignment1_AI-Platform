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

ly** using `.backward()`.  
