"""Teach a population of tiny neural networks to play the dino game.

Method: neuroevolution (a genetic algorithm over network weights).
  1. Create POP random networks.
  2. Every network plays the SAME level at the same time.
  3. Score = how many frames it survived (its "fitness").
  4. Keep the best few, breed + mutate them to make the next generation.
  5. Repeat. No gradients and no backprop, just survival of the fittest.

Usage:
    python train.py                  # train and watch it live
    python train.py --headless       # train fast with no window
    python train.py --watch          # watch the saved best network play

While watching:
    UP / DOWN   simulation speed (frames per screen update)
    F           fast-forward: skip drawing until this generation ends
    B           show / hide the brain panel
    ESC         quit (the best network is saved after every generation)
"""

import argparse

import numpy as np

from dino_game import GROUND_Y, HEIGHT, N_ACTIONS, WIDTH, DinoGame, draw_dino

N_IN = DinoGame.N_OBS
N_HIDDEN = 10
N_OUT = N_ACTIONS

POP = 150           # networks per generation
ELITES = 8          # best ones copied unchanged into the next generation
PARENTS = 25        # top-N that are allowed to have children
MUTATION_RATE = 0.15
MUTATION_STRENGTH = 0.4
MAX_FRAMES = 8000   # end a generation if someone survives this long

SAVE_PATH = "best_dino.npz"

INPUT_NAMES = ["dist 1", "width", "bottom", "top", "dist 2", "dino y", "dino vy", "speed"]
OUTPUT_NAMES = ["run", "jump", "duck"]


# --- The neural network ------------------------------------------------------------
# All POP networks are stored in stacked arrays so one numpy call runs all of them:
#   W1: (POP, N_HIDDEN, N_IN)  b1: (POP, N_HIDDEN)
#   W2: (POP, N_OUT, N_HIDDEN) b2: (POP, N_OUT)

def random_population(n, rng):
    return {
        "W1": rng.normal(0, 1, (n, N_HIDDEN, N_IN)),
        "b1": rng.normal(0, 1, (n, N_HIDDEN)),
        "W2": rng.normal(0, 1, (n, N_OUT, N_HIDDEN)),
        "b2": rng.normal(0, 1, (n, N_OUT)),
    }


def forward(pop, x):
    """x: (n, N_IN) -> hidden activations (n, N_HIDDEN), output scores (n, N_OUT)."""
    hidden = np.tanh(np.einsum("phi,pi->ph", pop["W1"], x) + pop["b1"])
    out = np.einsum("poh,ph->po", pop["W2"], hidden) + pop["b2"]
    return hidden, out


def next_generation(pop, fitness, rng):
    """Selection -> crossover -> mutation."""
    order = np.argsort(fitness)[::-1]
    parents = {k: v[order[:PARENTS]] for k, v in pop.items()}
    n_children = POP - ELITES

    child = {}
    mom = rng.integers(0, PARENTS, n_children)
    dad = rng.integers(0, PARENTS, n_children)
    for k, v in parents.items():
        # Crossover: each weight comes from mom or dad with 50/50 chance.
        pick_mom = rng.random((n_children,) + v.shape[1:]) < 0.5
        c = np.where(pick_mom, v[mom], v[dad])
        # Mutation: nudge a random subset of weights.
        mutate = rng.random(c.shape) < MUTATION_RATE
        c = c + mutate * rng.normal(0, MUTATION_STRENGTH, c.shape)
        child[k] = np.concatenate([v[:ELITES], c])
    return child


def save_best(pop, idx, path=SAVE_PATH):
    np.savez(path, **{k: v[idx:idx + 1] for k, v in pop.items()})


def load_best(path=SAVE_PATH):
    data = np.load(path)
    return {k: data[k] for k in ("W1", "b1", "W2", "b2")}


# --- Running a generation ------------------------------------------------------------
class Generation:
    """Every network gets its own copy of the same level (same seed)."""

    def __init__(self, pop, seed):
        self.pop = pop
        n = len(pop["W1"])
        self.games = [DinoGame(seed) for _ in range(n)]
        self.alive = np.ones(n, dtype=bool)
        self.fitness = np.zeros(n)
        self.obs = np.stack([g.observe() for g in self.games])
        self.hidden = np.zeros((n, N_HIDDEN))
        self.out = np.zeros((n, N_OUT))
        self.frames = 0

    @property
    def over(self):
        return not self.alive.any() or self.frames >= MAX_FRAMES

    def step(self):
        self.hidden, self.out = forward(self.pop, self.obs)
        actions = self.out.argmax(axis=1)
        for i in np.flatnonzero(self.alive):
            obs, _, done = self.games[i].step(actions[i])
            self.obs[i] = obs
            self.fitness[i] = self.games[i].score
            if done:
                self.alive[i] = False
        self.frames += 1

    def leader(self):
        """Index of the network we show in the brain panel."""
        alive = np.flatnonzero(self.alive)
        return alive[0] if len(alive) else int(self.fitness.argmax())


# --- Visualisation ---------------------------------------------------------------------
PANEL_H = 320
BG = (247, 247, 247)
INK = (83, 83, 83)
POS = (40, 110, 230)   # positive weight
NEG = (220, 60, 60)    # negative weight


class Viewer:
    def __init__(self):
        import pygame
        self.pg = pygame
        pygame.init()
        self.show_brain = True
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT + PANEL_H))
        pygame.display.set_caption("Dino AI - neuroevolution")
        self.font = pygame.font.SysFont(None, 22)
        self.big = pygame.font.SysFont(None, 28)
        self.clock = pygame.time.Clock()
        self.steps_per_frame = 1
        self.fast_forward = False

    def handle_events(self):
        """Returns False if the user wants to quit."""
        pg = self.pg
        for e in pg.event.get():
            if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                return False
            if e.type == pg.KEYDOWN:
                if e.key == pg.K_UP:
                    self.steps_per_frame = min(64, self.steps_per_frame * 2)
                elif e.key == pg.K_DOWN:
                    self.steps_per_frame = max(1, self.steps_per_frame // 2)
                elif e.key == pg.K_f:
                    self.fast_forward = True
                elif e.key == pg.K_b:
                    self.show_brain = not self.show_brain
                    extra = PANEL_H if self.show_brain else 0
                    self.screen = pg.display.set_mode((WIDTH, HEIGHT + extra))
        return True

    def text(self, s, pos, color=INK, font=None):
        self.screen.blit((font or self.font).render(s, True, color), pos)

    def draw(self, gen, generation, history, best_ever):
        pg, screen = self.pg, self.screen
        screen.fill(BG)
        lead = gen.leader()

        # --- game: every living dino at once, the leader drawn solid ---
        pg.draw.line(screen, INK, (0, GROUND_Y), (WIDTH, GROUND_Y), 2)
        game = gen.games[lead]
        for o in game.obstacles:
            color = (60, 140, 60) if o.kind == "cactus" else (120, 80, 160)
            pg.draw.rect(screen, color, (o.x, GROUND_Y - o.top, o.w, o.h))

        for i in np.flatnonzero(gen.alive):
            if i != lead:
                hue = (i * 47) % 200
                draw_dino(screen, gen.games[i], (hue, 120, 255 - hue, 45))
        draw_dino(screen, game)

        self.text(f"generation {generation}", (10, 10), font=self.big)
        self.text(f"alive {gen.alive.sum()}/{len(gen.alive)}   score {gen.frames}   "
                  f"best ever {best_ever}   sim x{self.steps_per_frame}", (10, 36))

        if self.show_brain:
            self.draw_brain(gen, lead, history)
        pg.display.flip()

    def draw_brain(self, gen, lead, history):
        pg, screen = self.pg, self.screen
        top = HEIGHT
        pg.draw.line(screen, (210, 210, 210), (0, top), (WIDTH, top), 2)
        self.draw_network(gen, lead, pg.Rect(10, top + 10, 480, PANEL_H - 20))
        self.draw_chart(history, pg.Rect(510, top + 30, 275, PANEL_H - 70))

    def draw_network(self, gen, i, area):
        pg, screen = self.pg, self.screen
        pop = gen.pop
        x_in, x_hid, x_out = area.x + 125, area.x + 285, area.x + 430

        def column(n, x):
            gap = area.height / n
            return [(x, int(area.y + gap * (k + 0.5))) for k in range(n)]

        p_in, p_hid, p_out = column(N_IN, x_in), column(N_HIDDEN, x_hid), column(N_OUT, x_out)

        # Weights: color = sign, thickness = size.
        def links(W, src, dst):
            for d, pd in enumerate(dst):
                for s, ps in enumerate(src):
                    w = W[d, s]
                    width = max(1, min(5, int(abs(w) * 2)))
                    pg.draw.line(screen, POS if w > 0 else NEG, ps, pd, width)

        links(pop["W1"][i], p_in, p_hid)
        links(pop["W2"][i], p_hid, p_out)

        # Neurons: brightness = activation.
        def neuron(p, a, label=None, label_left=True, highlight=False):
            a = float(np.clip(a, -1, 1))
            c = POS if a >= 0 else NEG
            fill = tuple(int(255 - (255 - ch) * abs(a)) for ch in c)
            pg.draw.circle(screen, fill, p, 11)
            pg.draw.circle(screen, (0, 0, 0) if highlight else INK, p, 11, 3 if highlight else 1)
            if label:
                img = self.font.render(label, True, INK)
                screen.blit(img, (p[0] - 18 - img.get_width(), p[1] - 7) if label_left
                            else (p[0] + 18, p[1] - 7))

        for k, p in enumerate(p_in):
            neuron(p, gen.obs[i, k], f"{INPUT_NAMES[k]} {gen.obs[i, k]:.2f}")
        for k, p in enumerate(p_hid):
            neuron(p, gen.hidden[i, k])
        out = gen.out[i]
        choice = int(out.argmax())
        for k, p in enumerate(p_out):
            neuron(p, np.tanh(out[k]), OUTPUT_NAMES[k], label_left=False,
                   highlight=k == choice)

    def draw_chart(self, history, area):
        pg, screen = self.pg, self.screen
        self.text("fitness per generation", (area.x, area.y - 22))
        pg.draw.rect(screen, (225, 225, 225), area, 1)
        if not history:
            self.text("(after generation 1)", (area.x + 10, area.y + 10))
            return
        best = [b for b, _ in history]
        avg = [a for _, a in history]
        top = max(best) or 1

        def line(vals, color):
            if len(vals) == 1:
                vals = vals * 2
            pts = [(area.x + area.width * k / (len(vals) - 1),
                    area.bottom - area.height * v / top) for k, v in enumerate(vals)]
            pg.draw.lines(screen, color, False, pts, 2)

        line(best, POS)
        line(avg, (240, 150, 40))
        self.text(f"best {best[-1]}", (area.x, area.bottom + 6), POS)
        self.text(f"avg {avg[-1]:.0f}", (area.x + 120, area.bottom + 6), (240, 150, 40))
        self.text(f"{top}", (area.right - 45, area.y + 4))


# --- Main loops ------------------------------------------------------------------------
def train(headless, generations, seed):
    rng = np.random.default_rng(seed)
    pop = random_population(POP, rng)
    viewer = None if headless else Viewer()
    history, best_ever = [], 0

    for generation in range(1, generations + 1):
        gen = Generation(pop, seed=int(rng.integers(1 << 30)))
        while not gen.over:
            if viewer and not viewer.fast_forward:
                for _ in range(viewer.steps_per_frame):
                    if not gen.over:
                        gen.step()
                if not viewer.handle_events():
                    return
                viewer.draw(gen, generation, history, best_ever)
                viewer.clock.tick(60)
            else:
                gen.step()
                if viewer and gen.frames % 200 == 0 and not viewer.handle_events():
                    return

        if viewer:
            viewer.fast_forward = False
        best = int(gen.fitness.max())
        history.append((best, float(gen.fitness.mean())))
        if best >= best_ever:
            best_ever = best
            save_best(pop, int(gen.fitness.argmax()))
        print(f"gen {generation:3d}  best {best:5d}  avg {gen.fitness.mean():7.1f}  "
              f"best ever {best_ever}")
        pop = next_generation(pop, gen.fitness, rng)

    print(f"Saved the best network to {SAVE_PATH}. Watch it with: python train.py --watch")


def watch():
    pop = load_best()
    viewer = Viewer()
    history, best_ever, run = [], 0, 0
    while True:
        run += 1
        gen = Generation(pop, seed=None)
        while not gen.over:
            for _ in range(viewer.steps_per_frame):
                if not gen.over:
                    gen.step()
            if not viewer.handle_events():
                return
            viewer.draw(gen, run, history, best_ever)
            viewer.clock.tick(60)
        best_ever = max(best_ever, int(gen.fitness[0]))
        history.append((int(gen.fitness[0]), float(gen.fitness[0])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", help="train without a window")
    ap.add_argument("--watch", action="store_true", help="watch the saved best network")
    ap.add_argument("--generations", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.watch:
        watch()
    else:
        train(args.headless, args.generations, args.seed)
