# 🧠 MNIST Training with PyTorch on macOS (MPS Backend)

This project implements a simple Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset using **PyTorch**. It is optimized to run on **macOS** devices using Apple’s **Metal Performance Shaders (MPS)** backend for GPU acceleration.

> ✅ Ideal for beginners who want to learn deep learning, experiment with PyTorch, and understand how to leverage hardware acceleration on macOS.

---

## 📌 Features

- ✅ Simple and readable CNN architecture
- ✅ Trains on CPU or MPS (Apple Silicon GPU)
- ✅ Modular code with training and evaluation loops
- ✅ MNIST dataset included via `torchvision`

---

## 🚀 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yuno-gen/mnist-pytorch-mps.git
cd mnist-pytorch-mps
```

### 2. Set up Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install torch torchvision
```

> ℹ️ Make sure you're using PyTorch version **1.12+** with MPS support. You can check compatibility here: [PyTorch on MPS](https://pytorch.org/docs/stable/notes/mps.html)

---

## 🏃‍♂️ Run the Project

```bash
python main.py
```

You should see training progress with loss values and final accuracy output.

---

## 🧠 Model Architecture

```text
Input → Conv2D → ReLU → Conv2D → ReLU → MaxPool → Dropout
→ Flatten → Linear → ReLU → Dropout → Linear → LogSoftmax
```

---

## 📈 Sample Output

```text
Epoch 1 [0/60000]  Loss: 2.3123
...
Test set: Avg loss: 0.0457, Accuracy: 9843/10000 (98.43%)
```

---

## 📚 Learning Goals

This project helps you:

- Understand CNNs in PyTorch
- Learn training loops and evaluation techniques
- Use Apple's MPS backend to accelerate training on macOS
- Prepare for more advanced topics like distributed training (FSDP, DDP)

---

## 💡 Next Steps

- Add `torch.fx` for model tracing and graph transformations
- Try deeper models (ResNet, etc.)
- Integrate TensorBoard or WandB for logging
- Experiment with quantization or pruning


