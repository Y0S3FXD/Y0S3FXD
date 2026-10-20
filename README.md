# Dino AI

A Chrome-style dinosaur game, plus a population of small neural networks that learn to play it.
While it trains you watch the AI play, and you can see its network work in real time.

```bash
pip install -r requirements.txt
python dino_game.py           # play it yourself (space = jump, hold for higher; down = duck)
python train.py               # watch the AI learn
python train.py --headless    # train fast in the terminal (about 5 s for 25 generations)
python train.py --watch       # watch the best saved network play
```

Training keys: `UP`/`DOWN` change simulation speed, `F` fast-forwards to the end of the generation, `B` shows or hides the brain panel, and `ESC` quits.

## The game

Plays like the Chrome offline dino game:
- **Gets harder:** speed climbs from 6 to 14, and gaps between obstacles scale with it.
- **Jump and duck:** tap jump for a small hop or hold it to jump higher. Down ducks, or drops you faster mid-air.
- **Obstacles:** cacti in groups of 1 to 3, and (after a while) birds with flapping wings at three heights: low (jump), middle (duck) and high (run under).
- **Looks and sound:** clouds, a scrolling ground, night mode with moon and stars every 700 points, a score that flashes every 100, a saved high score, X-eyes on game over, and generated sounds.

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
