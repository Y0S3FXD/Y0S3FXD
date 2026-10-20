"""Teach a population of tiny neural networks to play the dino game or Slope.

Method: neuroevolution (a genetic algorithm over network weights).
  1. Create POP random networks.
  2. Every network plays the SAME level at the same time.
  3. Score = how many frames it survived (its "fitness").
  4. Keep the best few, breed + mutate them to make the next generation.
  5. Repeat. No gradients and no backprop, just survival of the fittest.

Usage:
    python train.py                         # dino: train and watch it live
    python train.py --game slope            # Slope: train and watch it live
    python train.py --game slope --headless # train fast with no window
    python train.py --game slope --watch    # watch the saved best network play

While watching:
    UP / DOWN   simulation speed (frames per screen update)
    F           fast-forward: skip drawing until this generation ends
    B           show / hide the brain panel
    R           show / hide the AI's sight rays (Slope)
    ESC         quit (the best network is saved after every generation)
"""

import argparse
import os

import numpy as np

import dino_game
import slope_game

# Everything the trainer needs to know about each game.
GAMES = {
    "dino": dict(
        batch=dino_game.DinoBatch,
        renderer=dino_game.Renderer,
        size=(dino_game.WIDTH, dino_game.HEIGHT),
        inputs=["dist 1", "width", "bottom", "top", "dist 2", "dino y", "dino vy", "speed"],
        outputs=["run", "jump", "duck"],
        hidden=10,
        generations=50,
        max_frames=8000,
    ),
    "slope": dict(
        batch=slope_game.SlopeBatch,
        renderer=slope_game.SlopeRenderer,
        size=(slope_game.WIDTH, slope_game.HEIGHT),
        inputs=[f"ray {a:+d}°" for a in slope_game.RAY_ANGLES_DEG] + ["side vel", "speed"],
        outputs=["left", "straight", "right"],
        hidden=16,
        generations=100,
        max_frames=8000,
    ),
}

POP = 150           # networks per generation
ELITES = 8          # best ones copied unchanged into the next generation
PARENTS = 25        # top-N that are allowed to have children
MUTATION_RATE = 0.15
MUTATION_STRENGTH = 0.4


# --- The neural network ------------------------------------------------------------
# All POP networks are stored in stacked arrays so one numpy call runs all of them:
#   W1: (POP, n_hidden, n_in)  b1: (POP, n_hidden)
#   W2: (POP, n_out, n_hidden) b2: (POP, n_out)

def random_population(n, n_in, n_hidden, n_out, rng):
    return {
        "W1": rng.normal(0, 1, (n, n_hidden, n_in)),
        "b1": rng.normal(0, 1, (n, n_hidden)),
        "W2": rng.normal(0, 1, (n, n_out, n_hidden)),
        "b2": rng.normal(0, 1, (n, n_out)),
    }


def forward(pop, x):
    """x: (n, n_in) -> hidden activations (n, n_hidden), output scores (n, n_out)."""
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


def save_path(game):
    return f"best_{game}.npz"


def save_best(pop, idx, path):
    np.savez(path, **{k: v[idx:idx + 1] for k, v in pop.items()})


def load_best(path):
    data = np.load(path)
    return {k: data[k] for k in ("W1", "b1", "W2", "b2")}


# --- Running a generation ------------------------------------------------------------
class Generation:
    """Every network plays its own copy of the same level (same seed)."""

    def __init__(self, cfg, pop, seed):
        self.cfg = cfg
        self.pop = pop
        n = len(pop["W1"])
        self.env = cfg["batch"](n, seed)
        self.hidden = np.zeros((n, cfg["hidden"]))
        self.out = np.zeros((n, len(cfg["outputs"])))

    @property
    def alive(self):
        return self.env.alive

    @property
    def fitness(self):
        return self.env.scores

    @property
    def frames(self):
        return self.env.frames

    @property
    def over(self):
        return not self.alive.any() or self.frames >= self.cfg["max_frames"]

    def step(self):
        self.hidden, self.out = forward(self.pop, self.env.obs)
        self.env.step(self.out.argmax(axis=1))

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
    def __init__(self, cfg, title):
        import pygame
        self.pg = pygame
        pygame.init()
        self.cfg = cfg
        self.width, self.height = cfg["size"]
        self.show_brain = True
        self.screen = pygame.display.set_mode((self.width, self.height + PANEL_H))
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont(None, 22)
        self.big = pygame.font.SysFont(None, 28)
        self.clock = pygame.time.Clock()
        self.steps_per_frame = 1
        self.fast_forward = False
        self.renderer = cfg["renderer"](self.screen)
        self.hi_score = 0

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
                elif e.key == pg.K_r and hasattr(self.renderer, "show_rays"):
                    self.renderer.show_rays = not self.renderer.show_rays
                elif e.key == pg.K_b:
                    self.show_brain = not self.show_brain
                    extra = PANEL_H if self.show_brain else 0
                    self.screen = pg.display.set_mode((self.width, self.height + extra))
                    self.renderer.screen = self.screen
        return True

    def text(self, s, pos, color=INK, font=None):
        self.screen.blit((font or self.font).render(s, True, color), pos)

    def draw(self, gen, generation, history, label="generation"):
        pg, screen, r = self.pg, self.screen, self.renderer
        screen.fill(BG)
        lead = gen.leader()

        # --- game: every living player at once, the leader drawn solid ---
        r.update_population(gen.env, lead, self.steps_per_frame)
        self.hi_score = max(self.hi_score, int(gen.env.display_scores[lead]))
        screen.set_clip(pg.Rect(0, 0, self.width, self.height))  # keep it out of the panel
        r.draw_population(gen.env, lead, self.hi_score)
        screen.set_clip(None)

        self.text(f"{label} {generation}", (10, 10), r.ink, font=self.big)
        self.text(f"alive {gen.alive.sum()}/{len(gen.alive)}   sim x{self.steps_per_frame}",
                  (10, 36), r.ink)

        if self.show_brain:
            self.draw_brain(gen, lead, history)
        pg.display.flip()

    def draw_brain(self, gen, lead, history):
        pg, screen = self.pg, self.screen
        top = self.height
        pg.draw.line(screen, (210, 210, 210), (0, top), (self.width, top), 2)
        self.draw_network(gen, lead, pg.Rect(10, top + 10, 480, PANEL_H - 20))
        self.draw_chart(history, pg.Rect(510, top + 30, 275, PANEL_H - 70))

    def draw_network(self, gen, i, area):
        pg, screen = self.pg, self.screen
        pop = gen.pop
        x_in, x_hid, x_out = area.x + 125, area.x + 285, area.x + 430
        n_in, n_hid, n_out = pop["W1"].shape[2], pop["W1"].shape[1], pop["W2"].shape[1]
        r_node = 11 if n_hid <= 12 else 8

        def column(n, x):
            gap = area.height / n
            return [(x, int(area.y + gap * (k + 0.5))) for k in range(n)]

        p_in, p_hid, p_out = column(n_in, x_in), column(n_hid, x_hid), column(n_out, x_out)

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
            pg.draw.circle(screen, fill, p, r_node)
            pg.draw.circle(screen, (0, 0, 0) if highlight else INK, p, r_node,
                           3 if highlight else 1)
            if label:
                img = self.font.render(label, True, INK)
                screen.blit(img, (p[0] - 18 - img.get_width(), p[1] - 7) if label_left
                            else (p[0] + 18, p[1] - 7))

        obs = gen.env.obs
        names = self.cfg["inputs"]
        for k, p in enumerate(p_in):
            neuron(p, obs[i, k], f"{names[k]} {obs[i, k]:.2f}")
        for k, p in enumerate(p_hid):
            neuron(p, gen.hidden[i, k])
        out = gen.out[i]
        choice = int(out.argmax())
        for k, p in enumerate(p_out):
            neuron(p, np.tanh(out[k]), self.cfg["outputs"][k], label_left=False,
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
def train(game, headless, generations, seed):
    cfg = GAMES[game]
    path = save_path(game)
    rng = np.random.default_rng(seed)
    pop = random_population(POP, cfg["batch"].N_OBS, cfg["hidden"], len(cfg["outputs"]), rng)
    viewer = None if headless else Viewer(cfg, f"{game} AI - neuroevolution")
    history, best_ever = [], 0

    for generation in range(1, generations + 1):
        gen = Generation(cfg, pop, seed=int(rng.integers(1 << 30)))
        while not gen.over:
            if viewer and not viewer.fast_forward:
                for _ in range(viewer.steps_per_frame):
                    if not gen.over:
                        gen.step()
                if not viewer.handle_events():
                    return
                viewer.draw(gen, generation, history)
                viewer.clock.tick(60)
            else:
                gen.step()
                if viewer and gen.frames % 200 == 0 and not viewer.handle_events():
                    return

        if viewer:
            viewer.fast_forward = False
            viewer.hi_score = max(viewer.hi_score, int(gen.env.display_scores.max()))
        best = int(gen.fitness.max())
        history.append((best, float(gen.fitness.mean())))
        if best >= best_ever:
            best_ever = best
            save_best(pop, int(gen.fitness.argmax()), path)
        print(f"gen {generation:3d}  best {best:5d}  avg {gen.fitness.mean():7.1f}  "
              f"best ever {best_ever}")
        pop = next_generation(pop, gen.fitness, rng)

    print(f"Saved the best network to {path}. "
          f"Watch it with: python train.py --game {game} --watch")


def watch(game):
    cfg = GAMES[game]
    path = save_path(game)
    if not os.path.exists(path):
        print(f"No trained network yet ({path}). Train one first: python train.py --game {game}")
        return
    pop = load_best(path)
    viewer = Viewer(cfg, f"{game} AI - best network")
    history, run = [], 0
    while True:
        run += 1
        gen = Generation(cfg, pop, seed=None)
        while not gen.over:
            for _ in range(viewer.steps_per_frame):
                if not gen.over:
                    gen.step()
            if not viewer.handle_events():
                return
            viewer.draw(gen, run, history, label="run")
            viewer.clock.tick(60)
        history.append((int(gen.fitness[0]), float(gen.fitness[0])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", choices=GAMES, default="dino")
    ap.add_argument("--headless", action="store_true", help="train without a window")
    ap.add_argument("--watch", action="store_true", help="watch the saved best network")
    ap.add_argument("--generations", type=int, help="default depends on the game")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.watch:
        watch(args.game)
    else:
        train(args.game, args.headless,
              args.generations or GAMES[args.game]["generations"], args.seed)
