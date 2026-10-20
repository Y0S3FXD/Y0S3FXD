# Dino AI

A Chrome-style dinosaur game, plus a population of small neural networks that learn to play it.
While it trains you watch the AI play, and you can see its network work in real time.

```bash
pip install -r requirements.txt
python dino_game.py           # play it yourself (space = jump, down = duck)
python train.py               # watch the AI learn
python train.py --headless    # train fast in the terminal (about 5 s for 25 generations)
python train.py --watch       # watch the best saved network play
```

Training keys: `UP`/`DOWN` change simulation speed, `F` fast-forwards to the end of the generation, `B` shows or hides the brain panel, and `ESC` quits.

## What you're looking at

**Top:** all 150 dinos play the same level together. The solid one is the "leader" and the faint ones are the rest of the population. Watch them die off at each obstacle.

**Bottom left:** the leader's brain.
- **Inputs (left):** the 8 numbers it sees: distance to the next obstacle, its size and height, the next obstacle after that, the dino's own height and vertical speed, and the game speed.
- **Lines:** the weights. Blue is positive, red is negative, and thicker means a larger value.
- **Circles:** each neuron's activation, with brighter meaning stronger. Hidden neurons use `tanh`.
- **Outputs (right):** run / jump / duck. The one outlined in black has the highest score, and that's the action the dino takes.

**Bottom right:** best and average fitness (frames survived) for each generation.

## How it learns: neuroevolution

There's no backprop here. It uses a genetic algorithm over the weights:

1. Start with 150 networks that have random weights.
2. Fitness = how many frames each one survives.
3. Copy the top 8 into the next generation unchanged (elitism).
4. Make the rest by **crossover** (each weight comes from one of two top-25 parents) and **mutation** (random Gaussian nudges to about 15% of the weights).
5. Repeat.

Things to try for your course:
- Change `N_HIDDEN`, `POP`, `MUTATION_RATE`, or `MUTATION_STRENGTH` in `train.py` and compare the fitness curves.
- Remove an input from `observe()` in `dino_game.py`. Can it still learn?
- The next step is replacing the genetic algorithm with **DQN** (Q-learning with a neural net and backprop) on the same `DinoGame.step()` interface.
