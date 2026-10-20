# Dino AI and Slope AI

Two games, plus a population of small neural networks that learn to play them:
- the **Chrome dinosaur game**
- **Slope**, the neon 3D ball game

While it trains you watch the AI play, and you can see its network work in real time.

```bash
pip install -r requirements.txt

python dino_game.py                       # play the dino game yourself
python slope_game.py                      # play Slope yourself

python train.py                           # watch the AI learn the dino game
python train.py --game slope              # watch the AI learn Slope
python train.py --game slope --headless   # train fast in the terminal, no window
python train.py --game slope --watch      # watch the best saved network play
```

Training keys: `UP`/`DOWN` change simulation speed, `F` fast-forwards to the end of the generation, `B` shows or hides the brain panel, `R` shows or hides the AI's sight rays (Slope), and `ESC` quits.
The best network is saved after every generation, as `best_dino.npz` or `best_slope.npz`.

## The dino game

Plays like the Chrome offline dino game:
- **Gets harder:** speed climbs from 6 to 14, and gaps between obstacles scale with it.
- **Jump and duck:** tap jump for a small hop or hold it to jump higher. Down ducks, or drops you faster mid-air.
- **Obstacles:** cacti in groups of 1 to 3, and (after a while) birds with flapping wings at three heights: low (jump), middle (duck) and high (run under).
- **Looks and sound:** clouds, a scrolling ground, night mode with moon and stars every 700 points, a score that flashes every 100, a saved high score, X-eyes on game over, and generated sounds.

## Slope

A ball rolls down an endless neon track. Steer with `LEFT`/`RIGHT` (or `A`/`D`).
- **Gets harder:** speed climbs to about 2.5x the starting speed. The track curves more sharply, gets narrower (down to 3 tiles), and has more obstacles the further you go.
- **Ramps and jumps:** a ramp (yellow edges) launches the ball over a gap onto a lower platform. The platform may be shifted sideways, so you steer in the air to land on it. A glowing shadow shows where you'll come down.
- **Drops:** the track steps down a level, and the ball flies off the edge onto the lower part.
- **Moving blocks:** some red blocks slide from side to side.
- **Ways to die:** roll off the edge, fall through a hole, miss a landing, or smash into a red block.
- **Always beatable:**
  - Every obstacle leaves two neighbouring lanes open, and that opening is never further than the ball can reach in time, even at top speed.
  - Moving blocks never reach two neighbouring lanes.
  - Every jump can be cleared, because gravity bends the ball's path by the same amount per tile at any speed.
  - After each landing there's a stretch with no obstacles.
- **Looks:** a pseudo-3D downhill view, tiles with visible thickness, a rolling ball, falling and shattering animations, and a saved high score.

**Why it's harder than the dino game:** the dino only has to time a jump or duck. In Slope the AI has to choose a direction, commit to it, and plan ahead with momentum. Its steering is slidey, the track keeps moving under it, blocks move, and it has to aim its landings in mid-air.

## What you're looking at

**Top:** all 150 players play the same level together. The solid one is the "leader" and the faint ones are the rest of the population. Watch them die off at each obstacle.

**Bottom left:** the leader's brain.
- **Inputs (left):** the numbers it sees.
  - **Dino (8 inputs):** distance to the next obstacle, its size and height, the next obstacle after that, the dino's own height and vertical speed, and the game speed.
  - **Slope (9 inputs):** 7 **rays** fanned out in front of the ball (at -70° to +70°). Each one reports how far it gets before hitting an edge, a hole or a block (1.00 = clear for 10 tiles). The other two are the ball's sideways velocity and the game speed. The rays see straight across jump gaps, so in mid-air the AI can see where the landing platform is. The rays are drawn on the track in the training view: blue means clear, red means danger close.
- **Lines:** the weights. Blue is positive, red is negative, and thicker means a larger value.
- **Circles:** each neuron's activation, with brighter meaning stronger. Hidden neurons use `tanh`.
- **Outputs (right):** run / jump / duck (dino) or left / straight / right (Slope). The one outlined in black has the highest score, and that's the action taken.

**Bottom right:** best and average fitness (frames survived) for each generation.

## How it learns: neuroevolution

There's no backprop here. It uses a genetic algorithm over the weights:

1. Start with 150 networks that have random weights.
2. Fitness = how many frames each one survives.
3. Copy the top 8 into the next generation unchanged (elitism).
4. Make the rest by **crossover** (each weight comes from one of two top-25 parents) and **mutation** (random Gaussian nudges to about 15% of the weights).
5. Repeat.

The networks are tiny: 8 → 10 → 3 for the dino game and 9 → 16 → 3 for Slope. A generation ends when every player is dead or one survives 8000 frames.

Things to try for your course:
- Change `hidden` in the `GAMES` table in `train.py`, or `POP`, `MUTATION_RATE` or `MUTATION_STRENGTH`, and compare the fitness curves.
- Remove a ray from `RAY_ANGLES_DEG` in `slope_game.py`. How many rays does the AI really need?
- The next step is replacing the genetic algorithm with **DQN** (Q-learning with a neural net and backprop). It can use the same `step()` interface.
