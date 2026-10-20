"""A copy of Slope: a ball rolls down an endless neon track that keeps getting
faster. You can only steer left and right - there's no jump and no brake. The
ball only leaves the ground when it rolls off a ramp or an edge.

The track is built the way the real game builds it (see the Slope wiki,
slope.fandom.com/wiki/Level_Generation): pieces come in sections that grow
from 1 to 4 pieces, every section ends with a speedtunnel, and the pieces are
the real game's obstacles:
    early:   RNG blocks, slants, straights
    middle:  treblocks, tunnels, snakes
    late:    hors, verts
Thin pieces (straights and snakes) stop appearing after 50 points.

Like dino_game.py, the rules (SlopeBatch) have no graphics, so an AI can play
without a window. SlopeBatch runs n balls on the same track at once using
numpy, which keeps training fast.

Run `python slope_game.py` to play it yourself:
    LEFT / RIGHT (or A / D)  = steer
    SPACE                    = start / restart
    ESC                      = quit
"""

import os

import numpy as np

# --- World (units: 1 = one tile; x = sideways, y = up, z = down the slope) ----------
MAXW = 9                # widest a track row can be, in tiles
BALL_R = 0.25
HIT_R = 0.17            # a block this close to the ball's center counts as a hit
BLOCK_INSET = 0.08      # blocks are drawn this much smaller than their tile
BLOCK_H = 0.75          # a ball flying higher than this above the track clears a block

START_SPEED = 0.15      # tiles per frame
MAX_SPEED = 0.38
SPEED_UP = 0.00002      # a little faster every frame...
PAD_BOOST = 0.01        # ...and a lot faster on every speedtunnel's speed-up pad
STEER = 0.014           # sideways push per frame
FRICTION = 0.88         # share of sideways velocity kept each frame (slidey steering)
FALL_CURVE = 0.08       # gravity per tile travelled, so a jump always lands the same
                        # distance away, whatever the speed
SLANT = np.tan(np.radians(30))   # slants are tilted 30 degrees
SIDE_PULL = 0.0104      # sideways gravity on a slant, per frame and unit of tilt
RAMP_ROWS = 4
THIN_UNTIL = 500        # rows (50 points): no more thin pieces after this
HOR_FRAMES = 180        # a hor takes 3 s to slide across its platform
VERT_FRAMES = 120       # a vert takes 2 s to lift up and 2 s to come back down
VERT_LIFT = 2.75        # how high a vert lifts off the track: high enough that each
                        # one blocks only a fifth of the time, so with their 3 s and
                        # 6 s delays two of the three are always open
VERT_CLEAR = 2 * BALL_R + 0.05   # lifted at least this high, the ball rolls under it

FLOOR, HOLE, BLOCK = 0, 1, 2
HOR, VERT = 1, 2        # kinds of moving block

EARLY = ("rng_blocks", "slant", "straight")
MIDDLE = ("treblocks", "tunnel", "snake")
LATE = ("hors", "verts")
THIN = ("straight", "snake")

# Actions (ordered left to right so the brain panel reads naturally)
LEFT, NOOP, RIGHT = 0, 1, 2
N_ACTIONS = 3

# The AI "sees" with rays fanned out in front of the ball.
RAY_ANGLES_DEG = [-70, -40, -20, 0, 20, 40, 70]
RAY_ANGLES = np.radians(RAY_ANGLES_DEG)
RAY_MAX = 10.0
RAY_STEP = 0.4
RAY_T = np.arange(1, int(RAY_MAX / RAY_STEP) + 1) * RAY_STEP


class Track:
    """Endless track, built piece by piece from a seed.

    Row i covers z in [i, i+1). Along the row its left edge slides from left0 to
    left1 (curves), its height from h0 to h1 (ramps) and its sideways tilt from
    bank0 to bank1 (slants; a row tilts around its middle). Each row also has:
    width (tiles), tiles (FLOOR / HOLE / BLOCK per lane), gap (the empty rows
    after a ramp, which the ball flies over), pad (a speed-up pad), tube (inside
    a speedtunnel, where you can't fall off), and up to 3 moving blocks
    dyn[i, k] = (kind, a, b, c):
        HOR:  a block a tiles half-wide slides across the row, phase b frames
        VERT: lanes a .. a+b lift up and come back down, c frames behind
    """

    FIELDS = dict(left0=(), left1=(), h0=(), h1=(), bank0=(), bank1=(), width=(),
                  tiles=(MAXW,), gap=(), pad=(), tube=(), dyn=(3, 4))
    TYPES = dict(width=int, tiles=np.int8, gap=bool, pad=bool, tube=bool)

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.n = 0
        for f, shape in self.FIELDS.items():
            setattr(self, f, np.zeros((0,) + shape, self.TYPES.get(f, float)))
        self._queue = []            # rows planned but not built yet
        self._center, self._h = 0.0, 0.0
        self._section = 0
        for _ in range(25):         # a plain runway to start on
            self._emit(5)

    # -- building -----------------------------------------------------------------------
    def ensure(self, rows):
        """Make sure rows 0..rows-1 exist (built in chunks to avoid constant copying)."""
        if rows <= self.n:
            return
        new = []
        while self.n + len(new) < rows + 64:
            if not self._queue:
                self._plan_section(self.n + len(new))
            new.append(self._queue.pop(0))
        self.n += len(new)
        for f in self.FIELDS:
            values = np.array([r[f] for r in new], self.TYPES.get(f, float))
            setattr(self, f, np.concatenate([getattr(self, f), values]))

    def _plan_section(self, i):
        """Queue the next section: min(n, 4) pieces (an early one, a middle one,
        then late ones) and a speedtunnel to finish it, like the real game."""
        self._section += 1
        for tier in (EARLY, MIDDLE, LATE, LATE)[:min(self._section, 4)]:
            options = [p for p in tier if i < THIN_UNTIL or p not in THIN]
            getattr(self, "_" + str(self.rng.choice(options)))()
        self._speedtunnel()

    def _emit(self, width, tiles=None, h0=None, h1=None, bank=(0.0, 0.0), drift=0.0,
              gap=False, pad=False, tube=False, dyn=()):
        """Queue one row, `width` tiles wide around the current center line."""
        left0 = self._center - width / 2
        self._center += drift
        moving = np.zeros((3, 4))
        if dyn:
            moving[:len(dyn)] = dyn
        self._queue.append(dict(
            left0=left0, left1=self._center - width / 2,
            h0=self._h if h0 is None else h0, h1=self._h if h1 is None else h1,
            bank0=bank[0], bank1=bank[1], width=width,
            tiles=self._floor(width) if tiles is None else tiles,
            gap=gap, pad=pad, tube=tube, dyn=moving))

    @staticmethod
    def _floor(width):
        row = np.full(MAXW, HOLE, np.int8)
        row[:width] = FLOOR
        return row

    # -- the pieces (names from the Slope wiki) ----------------------------------------
    def _rng_blocks(self):
        """A short platform with one block on it, anywhere except the right edge."""
        w, j = 6, int(self.rng.integers(0, 4))       # the block covers lanes j, j + 1
        for k in range(14):
            row = self._floor(w)
            if k in (8, 9):
                row[j:j + 2] = BLOCK
            self._emit(w, row)

    def _slant(self):
        """A wide platform tilted 30 degrees to one side. Gravity pulls the ball
        towards the low edge, so you have to keep steering uphill."""
        w, b = 7, SLANT * self.rng.choice([-1, 1])
        twist = [b * k / 3 for k in range(4)]
        for k in range(3):
            self._emit(w, bank=(twist[k], twist[k + 1]))
        for _ in range(12):
            self._emit(w, bank=(b, b))
        for k in range(3, 0, -1):
            self._emit(w, bank=(twist[k], twist[k - 1]))

    def _straight(self):
        """A thin road that ends in a sudden drop."""
        for _ in range(4):
            self._emit(4)
        for _ in range(18):
            self._emit(2)
        self._h -= self.rng.uniform(1.5, 2.5)

    def _treblocks(self):
        """Blocks on both sides, then a bigger one in the middle behind them: go
        through the middle, then weave out through one of two skinny gaps."""
        w = 7
        for k in range(15):
            row = self._floor(w)
            if k in (4, 5):
                row[[0, 1, 5, 6]] = BLOCK
            if k in (9, 10):
                row[2:5] = BLOCK
            self._emit(w, row)

    def _tunnel(self):
        """Walls of red bricks on both sides, then a ramp at the end."""
        w = 5
        for _ in range(16):
            row = self._floor(w)
            row[[0, w - 1]] = BLOCK
            self._emit(w, row)
        self._ramp(w)

    def _snake(self):
        """A long, thin, curvy path that ends in a ramp."""
        for _ in range(4):
            self._emit(4)
        phase = self.rng.uniform(0, 2 * np.pi)
        for k in range(30):
            self._emit(2, drift=0.18 * np.sin(phase + 0.25 * k))
        self._ramp(2)

    def _hors(self):
        """Big blocks sliding from one side to the other, 3 s a crossing, each
        on its own random clock."""
        w = 6
        for k in range(18):
            dyn = [(HOR, 0.75, self.rng.uniform(0, 2 * HOR_FRAMES), 0)] if k in (4, 9, 14) else ()
            self._emit(w, dyn=dyn)

    def _verts(self):
        """Three blocks side by side that lift up (you can roll under them) and
        come back down, 2 s each way. The middle one runs 3 s behind the left
        one and the right one 6 s behind, so two of them are always open."""
        w, start = 6, self.rng.uniform(0, 4 * VERT_FRAMES)
        for k in range(12):
            dyn = [(VERT, 2 * v, 2, start + 180 * v) for v in range(3)] if k == 6 else ()
            self._emit(w, dyn=dyn)

    def _speedtunnel(self):
        """Ends every section: a round tunnel you can't fall out of, with a
        speed-up pad on the floor and a ramp that flies you to the next section."""
        w = 4
        for k in range(14):
            self._emit(w, tube=True, pad=k in (6, 7))
        self._ramp(w)

    def _ramp(self, w):
        """A ramp that launches the ball over a gap onto a lower platform. With
        the fixed fall curve the ball always flies at least 9 tiles, and the gap
        is at most 6, so the jump can always be made."""
        base, rise = self._h, self.rng.uniform(1.0, 1.4)
        for k in range(RAMP_ROWS):
            self._emit(w, h0=base + rise * k / RAMP_ROWS, h1=base + rise * (k + 1) / RAMP_ROWS)
        self._h = base - self.rng.uniform(0.0, 1.0)
        for _ in range(int(self.rng.integers(3, 7))):
            self._emit(0, gap=True)
        self._center += self.rng.uniform(-0.5, 0.5)

    # -- looking things up ------------------------------------------------------------
    def _rows(self, zs):
        zs = np.maximum(zs, 0.0)
        rows = zs.astype(int)
        if rows.size:
            self.ensure(int(rows.max()) + 2)
        return zs, rows

    def _along(self, a, b, zs):
        zs, rows = self._rows(zs)
        return a[rows] + (b[rows] - a[rows]) * (zs - rows)

    def left_at(self, zs):
        """x of the track's left edge at depth zs."""
        return self._along(self.left0, self.left1, zs)

    def height(self, zs):
        """Height of the middle of the track at depth zs (also over holes and gaps)."""
        return self._along(self.h0, self.h1, zs)

    def bank_at(self, zs):
        """Sideways tilt at depth zs: how much the track rises per tile to the right."""
        return self._along(self.bank0, self.bank1, zs)

    def surface(self, xs, zs):
        """Height of the track surface at (xs, zs), tilt included."""
        middle = self.left_at(zs) + self.width[self._rows(zs)[1]] / 2
        return self.height(zs) + self.bank_at(zs) * (xs - middle)

    @staticmethod
    def hor_x(dyn, width, t):
        """Where each hor's middle is at frame t, from its row's left edge."""
        half = dyn[..., 1]
        u = (t + dyn[..., 2]) / HOR_FRAMES
        return half + (width - 2 * half) * (1 - np.abs(u % 2 - 1))

    @staticmethod
    def vert_lift(dyn, t):
        """How high each vert is lifted off the track at frame t."""
        u = (t - dyn[..., 3]) / VERT_FRAMES
        return VERT_LIFT * (1 - np.abs(u % 2 - 1))

    def _look(self, xs, zs, t):
        zs, rows = self._rows(zs)
        rel = xs - self.left_at(zs)
        j = np.floor(rel).astype(int)
        inside = (j >= 0) & (j < MAXW)
        tile = np.where(inside, self.tiles[rows, np.clip(j, 0, MAXW - 1)], HOLE)
        dyn = self.dyn[rows]
        if (dyn[..., 0] > 0).any():
            rel, width = rel[..., None], self.width[rows][..., None]
            kind = dyn[..., 0]
            hor = (kind == HOR) & (np.abs(rel - self.hor_x(dyn, width, t)) < dyn[..., 1])
            vert = ((kind == VERT) & (rel >= dyn[..., 1]) & (rel < dyn[..., 1] + dyn[..., 2])
                    & (self.vert_lift(dyn, t) < VERT_CLEAR))
            tile = np.where((hor | vert).any(axis=-1) & (tile == FLOOR), BLOCK, tile)
        return tile, rows

    def at(self, xs, zs, t=0):
        """Tile type at world points (arrays of any shape) at frame t. Off the track is HOLE."""
        return self._look(xs, zs, t)[0]

    def danger(self, xs, zs, t=0):
        """What the AI's rays stop at: blocks, holes and edges. Jump gaps don't
        count, because the ball flies over them."""
        tile, rows = self._look(xs, zs, t)
        return (tile == BLOCK) | ((tile == HOLE) & ~self.gap[rows])


class SlopeBatch:
    """n balls on the same track, stepped together. n=1 for a human player."""

    N_OBS = len(RAY_ANGLES) + 3

    def __init__(self, n, seed=None):
        self.n = n
        self.track = Track(seed)
        self.x = np.zeros(n)
        self.vx = np.zeros(n)
        self.z = np.full(n, 1.0)
        self.y = np.zeros(n)                     # height of the bottom of the ball
        self.vy = np.zeros(n)                    # climb per tile travelled
        self.air = np.zeros(n, dtype=bool)       # flying (off a ramp or an edge)?
        self.speed = START_SPEED
        self.alive = np.ones(n, dtype=bool)
        self.scores = np.zeros(n, dtype=int)     # frames survived (fitness)
        self.death = np.zeros(n, dtype=np.int8)  # 0 alive, 1 fell, 2 hit a block
        self.frames = 0
        self._pad_row = -1
        self.obs = np.zeros((n, self.N_OBS), np.float32)
        self.observe(np.arange(n))

    @property
    def display_scores(self):
        return (self.z / 10).astype(int)         # points, like the real game's score

    def step(self, actions):
        idx = np.flatnonzero(self.alive)
        self.frames += 1
        if not idx.size:
            return
        track, dz, t = self.track, self.speed, self.frames

        # Steering, plus gravity pulling sideways on a slant.
        push = (actions[idx] == RIGHT).astype(float) - (actions[idx] == LEFT)
        pull = np.where(self.air[idx], 0.0, SIDE_PULL * track.bank_at(self.z[idx]))
        self.vx[idx] = (self.vx[idx] + push * STEER - pull) * FRICTION
        self.x[idx] += self.vx[idx]
        self.z[idx] += dz
        self.speed = min(MAX_SPEED, self.speed + SPEED_UP)
        self.scores[idx] += 1

        x, z = self.x[idx], self.z[idx]
        rows = z.astype(int)
        # In a speedtunnel the round walls keep the ball in.
        tube = track.tube[rows]
        if tube.any():
            left = track.left_at(z)
            lo, hi = left + BALL_R, left + track.width[rows] - BALL_R
            out = tube & ((x < lo) | (x > hi))
            x = self.x[idx] = np.where(tube, np.clip(x, lo, hi), x)
            self.vx[idx] = np.where(out, -0.3 * self.vx[idx], self.vx[idx])
        # The speed-up pad (all living balls are at the same depth).
        r = rows[0]
        if track.pad[r] and r != self._pad_row and not track.pad[r - 1]:
            self.speed = min(MAX_SPEED, self.speed + PAD_BOOST)
            self._pad_row = r

        under = track.at(x, z, t)
        surf = track.surface(x, z)
        # Fly along a curve; if that would go into the floor, roll on the floor instead.
        # A rolling ball also follows small dips (sliding down a slant) instead of
        # floating off them; only real edges like cliffs and ramp ends launch it.
        y = self.y[idx] + self.vy[idx] * dz - 0.5 * FALL_CURVE * dz * dz
        land = (under != HOLE) & ((y <= surf) | (~self.air[idx] & (y < surf + 0.15)))
        self.y[idx] = np.where(land, surf, y)
        self.vy[idx] = np.where(land, track.h1[rows] - track.h0[rows],
                                self.vy[idx] - FALL_CURVE * dz)
        self.air[idx] = ~land

        fell = (under == HOLE) & (self.y[idx] < surf + 0.15)
        low = self.y[idx] < surf + BLOCK_H
        crashed = low & ((under == BLOCK) | (track.at(x - HIT_R, z, t) == BLOCK) |
                         (track.at(x + HIT_R, z, t) == BLOCK) |
                         (track.at(x, z + HIT_R, t) == BLOCK))
        self.death[idx[fell]] = 1
        self.death[idx[crashed]] = 2
        self.alive[idx[fell | crashed]] = False
        self.observe(np.flatnonzero(self.alive))

    def ray_distances(self, idx):
        """(len(idx), 7): how far each ray gets before hitting an edge, hole or block."""
        xs = self.x[idx, None, None] + np.sin(RAY_ANGLES)[None, :, None] * RAY_T
        zs = self.z[idx, None, None] + np.cos(RAY_ANGLES)[None, :, None] * RAY_T
        bad = self.track.danger(xs, zs, self.frames)
        return np.where(bad.any(axis=2), RAY_T[bad.argmax(axis=2)], RAY_MAX)

    def observe(self, idx):
        """Update what balls idx see: 7 ray distances, sideways velocity, speed and
        the slant under the ball (rays can't see a tilt). Only living balls are
        updated, which keeps late generations fast."""
        n_rays = len(RAY_ANGLES)
        self.obs[idx, :n_rays] = self.ray_distances(idx) / RAY_MAX
        self.obs[idx, n_rays] = self.vx[idx] / 0.12
        self.obs[idx, n_rays + 1] = self.speed / MAX_SPEED
        self.obs[idx, n_rays + 2] = self.track.bank_at(self.z[idx]) / SLANT


# --- Drawing (pygame, pseudo-3D) ------------------------------------------------------
WIDTH, HEIGHT = 800, 450
SKY_LINE = 105          # screen y where the sky ends
VANISH_Y = 150          # where the track vanishes; below the sky line, so it looks downhill
FOCAL = 330             # bigger = more zoomed in
CAM_H = 1.5             # camera height above the ball
CAM_BACK = 3.0          # camera distance behind the ball
NEAR = 0.6              # nothing closer than this to the camera is drawn
VIEW_ROWS = 45
THICKNESS = 0.35        # how thick the track tiles are

SKY_TOP, SKY_BOTTOM, VOID = (4, 4, 18), (34, 12, 66), (2, 2, 10)
TILE, TILE_EDGE, TILE_SIDE = (8, 40, 22), (0, 255, 120), (0, 90, 50)
RAMP_EDGE, PAD, RING = (190, 255, 90), (255, 210, 40), (60, 200, 255)
BLOCK_TOP, BLOCK_FRONT, BLOCK_SIDE = (255, 80, 100), (205, 20, 45), (150, 10, 30)
BALL, BALL_DARK, BALL_LIGHT = (40, 255, 150), (10, 150, 80), (210, 255, 230)
HUD = (200, 255, 225)


def lerp(a, b, t):
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


class SlopeRenderer:
    def __init__(self, screen):
        import pygame
        self.pg = pygame
        self.screen = screen
        self.font = pygame.font.SysFont("couriernew,dejavusansmono,monospace", 22, bold=True)
        self.ink = HUD
        self.cam_x, self.cam_y, self.cam_z = 0.0, CAM_H, 0.0
        self.fall_t = 0         # frames since the followed ball died
        self.show_rays = True

        self.sky = pygame.Surface((WIDTH, HEIGHT))
        self.sky.fill(VOID)
        for y in range(SKY_LINE + 40):
            color = (lerp(SKY_TOP, SKY_BOTTOM, y / SKY_LINE) if y <= SKY_LINE
                     else lerp(SKY_BOTTOM, VOID, (y - SKY_LINE) / 40))
            self.sky.fill(color, (0, y, WIDTH, 1))
        rng = np.random.default_rng(1)
        for _ in range(60):
            x, y = rng.integers(0, WIDTH), rng.integers(0, SKY_LINE - 10)
            self.sky.fill((180, 180, 230), (x, y, 2, 2))

    def project(self, x, z, y=0.0):
        s = FOCAL / (z - self.cam_z)
        return (WIDTH / 2 + (x - self.cam_x) * s, VANISH_Y + (self.cam_y - y) * s)

    def update_population(self, batch, lead, frames=1):
        if batch.alive[lead]:
            # The camera glides after the ball, like the real game.
            self.cam_x += (batch.x[lead] - self.cam_x) * min(1.0, 0.12 * frames)
            self.cam_y += (batch.y[lead] + CAM_H - self.cam_y) * min(1.0, 0.2 * frames)
            self.fall_t = 0
        else:
            self.fall_t += frames

    def ball_pos(self, batch, i):
        """Where to draw ball i as (x, z, y of its bottom). A ball that fell keeps
        rolling forward as it drops."""
        x, z, y = batch.x[i], batch.z[i], batch.y[i]
        if batch.alive[i] or batch.death[i] != 1:
            return x, z, y
        t = self.fall_t
        dz = batch.speed * t
        return (x + batch.vx[i] * t, z + dz,
                y + batch.vy[i] * dz - 0.5 * FALL_CURVE * dz * dz - 0.01 * t * t)

    # -- drawing ----------------------------------------------------------------------
    def draw_population(self, batch, lead, hi_score=0):
        track = batch.track
        self.cam_z = batch.z[lead] - CAM_BACK
        near = self.cam_z + NEAR
        track.ensure(int(self.cam_z + VIEW_ROWS) + 2)
        self.screen.blit(self.sky, (0, 0))

        # Balls to draw (every living one, plus the leader), grouped by row.
        by_row = {}
        for i in [i for i in np.flatnonzero(batch.alive) if i != lead] + [lead]:
            x, z, y = self.ball_pos(batch, i)
            by_row.setdefault(int(z), []).append((i, x, z, y))

        # Painter's algorithm: far rows first, near rows last.
        for r in range(int(self.cam_z + VIEW_ROWS), max(int(near), 0) - 1, -1):
            z0, z1 = max(r, near), r + 1
            if z1 <= near:
                continue
            fog = min(1.0, (z0 - self.cam_z) / VIEW_ROWS) ** 1.5
            balls = by_row.get(r, [])
            sunk = [b[3] < track.surface(b[1], b[2]) - 0.05 for b in balls]
            for b, s in zip(balls, sunk):  # balls that fell below the track go under the tiles
                if s:
                    self.draw_ball(batch, *b, b[0] == lead)
            self.draw_row(track, r, z0, z1, fog, batch.frames)
            for b, s in zip(balls, sunk):
                if not s:
                    self.draw_ball(batch, *b, b[0] == lead)

        if self.show_rays and batch.alive[lead]:
            self.draw_rays(batch, lead)
        self.draw_score(batch, lead, hi_score)

    def draw_row(self, track, r, z0, z1, fog, t):
        w = int(track.width[r])
        if w == 0:
            return
        pg, p, screen = self.pg, self.project, self.screen
        tiles = track.tiles[r]
        ln, lf = float(track.left_at(z0)), float(track.left1[r])    # left edge, near/far end
        hn, hf = float(track.height(z0)), float(track.h1[r])        # height of the middle
        bn, bf = float(track.bank_at(z0)), float(track.bank1[r])    # tilt

        def yn(x):  # surface height along the near edge
            return hn + bn * (x - ln - w / 2)

        def yf(x):  # ... and along the far edge
            return hf + bf * (x - lf - w / 2)

        near_pts = [p(ln + k, z0, yn(ln + k)) for k in range(MAXW + 1)]
        far_pts = [p(lf + k, z1, yf(lf + k)) for k in range(MAXW + 1)]
        ramp = hf > float(track.h0[r]) + 1e-6
        top = lerp(TILE, VOID, fog)
        edge = lerp(RAMP_EDGE if ramp else TILE_EDGE, SKY_BOTTOM, fog)
        side, T = lerp(TILE_SIDE, VOID, fog), THICKNESS
        # The front edge shows its thickness where the row in front is missing or higher.
        if r > 0 and z0 == r:
            front_open = (track.tiles[r - 1] == HOLE) | (track.h1[r - 1] > hn + 0.01)
        else:
            front_open = np.zeros(MAXW, bool)

        for j in np.flatnonzero(tiles != HOLE):
            # The tile's thickness shows where it borders a hole or the track edge.
            xn, xf = ln + j, lf + j
            if (j == 0 or tiles[j - 1] == HOLE) and xn > self.cam_x:
                pg.draw.polygon(screen, side, [p(xf, z1, yf(xf)), p(xn, z0, yn(xn)),
                                               p(xn, z0, yn(xn) - T), p(xf, z1, yf(xf) - T)])
            if (j == MAXW - 1 or tiles[j + 1] == HOLE) and xn + 1 < self.cam_x:
                pg.draw.polygon(screen, side, [p(xf + 1, z1, yf(xf + 1)), p(xn + 1, z0, yn(xn + 1)),
                                               p(xn + 1, z0, yn(xn + 1) - T),
                                               p(xf + 1, z1, yf(xf + 1) - T)])
            if front_open[j]:
                pg.draw.polygon(screen, side, [p(xn, z0, yn(xn)), p(xn + 1, z0, yn(xn + 1)),
                                               p(xn + 1, z0, yn(xn + 1) - T), p(xn, z0, yn(xn) - T)])
            quad = [far_pts[j], far_pts[j + 1], near_pts[j + 1], near_pts[j]]
            pg.draw.polygon(screen, top, quad)
            pg.draw.polygon(screen, edge, quad, 1)

        h, left = (float(track.h0[r]) + hf) / 2, float(track.left_at(r + 0.5))
        if track.pad[r]:  # a speed-up arrow on the floor
            c = left + w / 2
            arrow = [p(c, min(r + 0.9, z1), h), p(c + 0.7, max(r + 0.2, z0), h),
                     p(c - 0.7, max(r + 0.2, z0), h)]
            pg.draw.polygon(screen, lerp(PAD, VOID, fog), arrow)
        for j in np.flatnonzero(tiles == BLOCK):
            self.draw_box(left + j + BLOCK_INSET, left + j + 1 - BLOCK_INSET, r, h, h + BLOCK_H, fog)
        for kind, a, b, c in track.dyn[r]:
            if kind == HOR:
                x = left + float(Track.hor_x(np.array([kind, a, b, c]), w, t))
                self.draw_box(x - a + BLOCK_INSET, x + a - BLOCK_INSET, r, h, h + 0.9, fog)
            elif kind == VERT:
                lift = h + float(Track.vert_lift(np.array([kind, a, b, c]), t))
                self.draw_box(left + a + BLOCK_INSET, left + a + b - BLOCK_INSET, r,
                              lift, lift + 0.8, fog)
        if track.tube[r] and z0 == r:  # the speedtunnel's round wall
            c, rad = left + w / 2, w / 2 + 0.15
            arch = [p(c + rad * np.cos(a), r, h + rad * np.sin(a)) for a in np.linspace(0, np.pi, 13)]
            pg.draw.lines(screen, lerp(RING, VOID, fog), False, arch, 2)

    def draw_box(self, x0, x1, r, y0, y1, fog):
        """A red block on row r, from x0 to x1 and from height y0 up to y1."""
        pg, p = self.pg, self.project
        zb0, zb1 = r + BLOCK_INSET, r + 1 - BLOCK_INSET
        if zb0 <= self.cam_z + NEAR:
            return
        if y1 < self.cam_y:      # we look down on its top...
            pg.draw.polygon(self.screen, lerp(BLOCK_TOP, VOID, fog),
                            [p(x0, zb1, y1), p(x1, zb1, y1), p(x1, zb0, y1), p(x0, zb0, y1)])
        elif y0 > self.cam_y:    # ...or up at its bottom
            pg.draw.polygon(self.screen, lerp(BLOCK_SIDE, VOID, fog),
                            [p(x0, zb1, y0), p(x1, zb1, y0), p(x1, zb0, y0), p(x0, zb0, y0)])
        side_x = x1 if x1 < self.cam_x else x0 if x0 > self.cam_x else None
        if side_x is not None:
            side = [p(side_x, zb1, y1), p(side_x, zb0, y1), p(side_x, zb0, y0), p(side_x, zb1, y0)]
            pg.draw.polygon(self.screen, lerp(BLOCK_SIDE, VOID, fog), side)
        front = [p(x0, zb0, y1), p(x1, zb0, y1), p(x1, zb0, y0), p(x0, zb0, y0)]
        pg.draw.polygon(self.screen, lerp(BLOCK_FRONT, VOID, fog), front)
        pg.draw.polygon(self.screen, lerp(BLOCK_TOP, VOID, fog), front, 1)

    def draw_ball(self, batch, i, x, z, y, is_lead):
        pg, screen = self.pg, self.screen
        dz = z - self.cam_z
        if dz < 0.8:
            return
        cx, cy = self.project(x, z, y + BALL_R)
        rad = max(2, int(BALL_R * FOCAL / dz))
        if not is_lead:
            ghost = pg.Surface((rad * 2, rad * 2), pg.SRCALPHA)
            hue = (i * 47) % 200
            pg.draw.circle(ghost, (hue, 255 - hue // 2, 255, 45), (rad, rad), rad)
            screen.blit(ghost, (cx - rad, cy - rad))
            return
        if batch.death[i] == 2:
            self.draw_burst(cx, cy, rad)
            return

        surf = float(batch.track.surface(x, z))
        if batch.alive[i] and y > surf + 0.05 and batch.track.at(np.array([x]), np.array([z]),
                                                                  batch.frames)[0] != HOLE:
            # A shadow with a glowing ring under a flying ball shows where it will land.
            sx, sy = self.project(x, z, surf)
            shadow = pg.Surface((rad * 2, rad), pg.SRCALPHA)
            pg.draw.ellipse(shadow, (0, 0, 0, 170), (0, 0, rad * 2, rad))
            pg.draw.ellipse(shadow, (*BALL, 160), (0, 0, rad * 2, rad), 2)
            screen.blit(shadow, (sx - rad, sy - rad / 2))

        glow = pg.Surface((rad * 4, rad * 4), pg.SRCALPHA)
        for k, alpha in ((2.0, 25), (1.5, 45)):
            pg.draw.circle(glow, (*BALL, alpha), (rad * 2, rad * 2), int(rad * k))
        screen.blit(glow, (cx - rad * 2, cy - rad * 2))
        pg.draw.circle(screen, BALL, (cx, cy), rad)
        # Stripes that roll up over the ball so you can see it spin.
        for k in range(3):
            a = (z * 1.5 + k * 2 * np.pi / 3) % (2 * np.pi)
            if a < np.pi:
                off = rad * np.cos(a)
                half = 0.9 * np.sqrt(max(rad * rad - off * off, 0.0))
                pg.draw.line(screen, BALL_DARK, (cx - half, cy + off), (cx + half, cy + off), 2)
        pg.draw.circle(screen, BALL_LIGHT, (cx - rad * 0.35, cy - rad * 0.35), rad * 0.3)

    def draw_burst(self, cx, cy, rad):
        """The ball shatters when it hits a block."""
        t = self.fall_t
        if t > 30:
            return
        for k in range(10):
            a = k * 2 * np.pi / 10 + 0.3
            d = rad * (0.5 + 0.25 * t)
            pos = (cx + np.cos(a) * d, cy + np.sin(a) * d * 0.6 + 0.08 * t * t)
            size = max(1, int(rad * 0.35 * (1 - t / 30)))
            self.pg.draw.circle(self.screen, lerp(BALL, BLOCK_TOP, t / 30), pos, size)

    def draw_rays(self, batch, i):
        """What the AI sees: a line per ray, blue = clear, red = danger close."""
        pg, track = self.pg, batch.track
        dists = batch.obs[i, :len(RAY_ANGLES)] * RAY_MAX
        x, z = batch.x[i], batch.z[i]
        start = self.project(x, z, batch.y[i] + 0.05)
        for ang, d in zip(RAY_ANGLES, dists):
            ex, ez = x + np.sin(ang) * d, z + np.cos(ang) * d
            end = self.project(ex, ez, float(track.surface(ex, ez)) + 0.05)
            color = lerp((255, 40, 60), (80, 255, 255), d / RAY_MAX)
            pg.draw.line(self.screen, color, start, end, 2)
            if d < RAY_MAX:
                pg.draw.circle(self.screen, color, end, 4)

    def draw_score(self, batch, lead, hi_score=0):
        s = int(batch.display_scores[lead])
        text = (f"HI {hi_score:05d}  " if hi_score else "") + f"{s:05d}"
        img = self.font.render(text, True, HUD)
        self.screen.blit(img, (WIDTH - img.get_width() - 15, 12))
        spd = self.font.render(f"speed {batch.speed / START_SPEED:.1f}x", True, HUD)
        self.screen.blit(spd, (WIDTH - spd.get_width() - 15, 38))

    def draw_game_over(self):
        img = self.font.render("G A M E   O V E R", True, (255, 80, 100))
        self.screen.blit(img, (WIDTH // 2 - img.get_width() // 2, HEIGHT // 2 - 60))


HI_SCORE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "slope_highscore.txt")


def play_human():
    import pygame
    from dino_game import Sounds, load_hi_score, save_hi_score

    pygame.mixer.pre_init(44100, -16, 1, 512)
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Slope")
    clock = pygame.time.Clock()
    renderer = SlopeRenderer(screen)
    renderer.show_rays = False
    sounds = Sounds()
    hint = pygame.font.SysFont(None, 28)

    game = SlopeBatch(1)
    hi_score = load_hi_score(HI_SCORE_FILE)
    started = False

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                return
            if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                if not started:
                    started = True
                elif not game.alive[0]:
                    game = SlopeBatch(1)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action = LEFT
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action = RIGHT
        else:
            action = NOOP

        if started and game.alive[0]:
            before, was_flying = game.display_scores[0], game.air[0]
            game.step(np.array([action]))
            if game.air[0] and not was_flying and game.alive[0]:
                sounds.play("jump")
            if game.display_scores[0] // 10 > before // 10:
                sounds.play("point")
            if not game.alive[0]:
                sounds.play("die")
                if game.display_scores[0] > hi_score:
                    hi_score = int(game.display_scores[0])
                    save_hi_score(hi_score, HI_SCORE_FILE)

        renderer.update_population(game, 0)
        renderer.draw_population(game, 0, hi_score)
        if not game.alive[0]:
            renderer.draw_game_over()
        if not started or not game.alive[0]:
            msg = ("press SPACE to start  -  steer with LEFT/RIGHT or A/D" if not started
                   else "press SPACE to restart")
            img = hint.render(msg, True, HUD)
            screen.blit(img, (WIDTH // 2 - img.get_width() // 2, HEIGHT // 2 - 20))
        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    play_human()
