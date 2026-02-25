# Implementation of **"D2G-DCIM: A Deep Reinforcement Learning and Graph Neural Network-Driven Model for Dynamic Competitive Influence Maximization"**

## Overview

This repository provides the implementation of the paper **"D2G-DCIM: A Deep Reinforcement Learning and Graph Neural Network-Driven Model for Dynamic Competitive Influence Maximization."**

## Project Structure

```
D2G-DCIM/
├── train_graphs/           # Training graph datasets
│   ├── graph0.txt
│   ├── graph1.txt
│   └── ... (10 graphs total)
├── main.py                 # Main training script
├── agent.py               # RL agent implementation (DDQN)
├── environment.py         # Graph environment and reward computation
├── models.py              # Neural network architectures (GNN blocks)
├── utils.py               # Utility functions and DCIC implementation
├── graph_process.py       # Graph processing
├── generate_train_graphs.py # Training data generation
├── test.py                # Model testing and evaluation
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Training Data

```bash
python generate_train_graphs.py
```

This generates 10 synthetic temporal graphs in the `train_graphs/` directory.

### 3. Train the Model

```bash
python main.py --num_epochs 2000 --k 5 --lr 5e-4
```

**Key Training Arguments:**
- `--num_epochs`: Number of training epochs (default: 2000)
- `--k`: Seed set size (default: 5)
- `--lr`: Learning rate (default: 5e-4)
- `--gamma`: Discount factor (default: 0.99)
- `--buffer_size`: Experience replay buffer size (default: 30000)
- `--test_graph_name`: Test graph name (default: 'Hypertext')

### 4. Test and Evaluate

```bash
python test.py
```

Configure test parameters in the script:
- `GRAPH_NAME`: Test graph name
- `SEED_BUDGET`: Number of seeds to select
- `S_B_K`: Competitor seed count
- `MODEL_PATH`: Trained model checkpoint path

## Citation

If you use this code in your research, please cite the original paper (citation to be added when available).

## License

This project is distributed under the MIT License. See LICENSE file for details.

---

*Note: Ensure all dependencies are installed and GPU is available for optimal training performance.*
