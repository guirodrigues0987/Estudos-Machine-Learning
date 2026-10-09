# Super Mario Bros Neuroevolution with NEAT-Python

[![CI](https://github.com/guirodrigues0987/Estudos-Machine-Learning/actions/workflows/ci.yml/badge.svg)](https://github.com/guirodrigues0987/Estudos-Machine-Learning/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This project implements an AI that learns to play Super Mario Bros (NES) using the
**NEAT** algorithm (NeuroEvolution of Augmenting Topologies). The main focus is
optimizing neural network topologies through evolutionary processes.

---

## Overview

Unlike traditional Deep Learning approaches, this project uses **genetic algorithms**
to evolve both the structure and the weights of a neural network.

### Key technical points

- **Hardware efficiency:** built to run in resource-constrained environments (tested on Lubuntu/Linux).
- **Data persistence:** a custom *checkpoint* system makes training resilient to crashes and power loss.
- **Engineering approach:** the *fitness* function rewards horizontal progress and a time bonus, which speeds up convergence.

---

## Tech stack

- **Language:** Python 3.11
- **Environment:** [Gym](https://github.com/openai/gym) (RL interface)
- **Emulator:** `gym-super-mario-bros`
- **Algorithm:** `neat-python` (neuroevolution)
- **Processing:** OpenCV (frame resizing and grayscale conversion)

---

## Results and evolution

Training runs in generations; the population size is set by `pop_size` in
`config-feedforward` (currently `5`, small enough to experiment on modest hardware).

- **Generation 0:** random movements.
- **Generation 20+:** first jumping patterns over obstacles and enemies (expected).
- **Generation 100+:** full mastery of level 1-1 (projection).

> **Engineering note:** the script saves the state of the evolution at generation 0
> (`checkpoint_generation_zero`) and every 5 generations afterwards
> (`checkpoint_latest`). Re-running the script resumes from the latest checkpoint.
> Checkpoints are pickle files and are not versioned - only load ones you created yourself.

---

## How to run

1. Clone the repository:
   ```bash
   git clone https://github.com/guirodrigues0987/Estudos-Machine-Learning.git
   cd Estudos-Machine-Learning
   ```
2. Create a virtual environment and install the dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Start training (opens the emulator window):
   ```bash
   python mario_evolutivo.py
   ```

When training finishes, the best genome is saved to `best_mario.pkl`.

---

## Project layout

```
mario_evolutivo.py   # Training script (evaluation, checkpointing, main loop)
config-feedforward   # NEAT configuration (population, mutation rates, network shape)
requirements.txt     # Pinned dependencies
```

## License

[MIT](LICENSE)
