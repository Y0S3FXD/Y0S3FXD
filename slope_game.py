"""A Slope-style game: a ball rolls down an endless neon track that keeps
getting faster, curvier and narrower. Steer around red blocks (some slide from
side to side), fly off ramps over gaps, drop down to lower platforms, and don't
fall through holes or off the edge.

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
SPEED_UP = 0.00004
STEER = 0.014           # sideways push per frame
FRICTION = 0.88         # share of sideways velocity kept each frame (slidey steering)
FALL_CURVE = 0.08       # gravity per tile travelled, so a jump always lands the same
                        # distance away, whatever the speed
HARD_AT = 1500          # rows until the track reaches full difficulty
FEATURES_FROM = 120     # rows before ramps, drops and moving blocks start
RAMP_ROWS = 4

FLOOR, HOLE, BLOCK = 0, 1, 2

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
    """Endless track, built row by row from a seed.

    Row i covers z in [i, i+1). Along the row its left edge slides from left0[i]
    to left1[i] (curves) and its height from h0[i] to h1[i] (ramps). gap[i] marks
    the empty rows after a ramp that the ball flies over. A row can also have a
    block sliding from side to side: mover[i] = (center, amplitude, speed, phase).
    """

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.n = 0
        self.left0, self.left1 = np.zeros(0), np.zeros(0)
        self.h0, self.h1 = np.zeros(0), np.zeros(0)
        self.tiles = np.zeros((0, MAXW), np.int8)     # FLOOR / HOLE / BLOCK
        self.gap = np.zeros(0, bool)
        self.mover = np.zeros((0, 4))
        self._queue = []            # rows planned but not built yet
        self._left, self._width, self._drift, self._h = -2.5, 5, 0.0, 0.0
        self._section = 30          # rows until the curve/width changes
        self._next_obstacle = 40    # rows until the next obstacle row
        self._next_feature = 60     # rows until the next ramp/drop/moving block
        self._quiet = 0             # rows kept clear after a landing
        self._hole_rows, self._hole_lanes = 0, None
        self._prev_free = None      # first of the two open lanes at the last obstacle
        self._since_obstacle = 0    # rows since then

    # -- building -----------------------------------------------------------------------
    def ensure(self, rows):
        """Make sure rows 0..rows-1 exist (built in chunks to avoid constant copying)."""
        if rows <= self.n:
            return
        new = []
        while self.n + len(new) < rows + 64:
            if not self._queue:
                self._plan(self.n + len(new))
            new.append(self._queue.pop(0))
        self.n += len(new)
        self.left0 = np.concatenate([self.left0, [r["left0"] for r in new]])
        self.left1 = np.concatenate([self.left1, [r["left1"] for r in new]])
        self.h0 = np.concatenate([self.h0, [r["h0"] for r in new]])
        self.h1 = np.concatenate([self.h1, [r["h1"] for r in new]])
        self.tiles = np.concatenate([self.tiles, np.array([r["tiles"] for r in new])])
        self.gap = np.concatenate([self.gap, [r["gap"] for r in new]])
        self.mover = np.concatenate([self.mover, np.array([r["mover"] for r in new])])

    def _emit(self, tiles, h0=None, h1=None, gap=False, mover=(0.0, 0.0, 0.0, 0.0)):
        """Queue one row, curving the track by the current drift."""
        left0 = self._left
        self._left += self._drift
        self._since_obstacle += 1
        self._queue.append(dict(left0=left0, left1=self._left, tiles=tiles,
                                h0=self._h if h0 is None else h0,
                                h1=self._h if h1 is None else h1,
                                gap=gap, mover=mover))

    def _floor(self):
        row = np.full(MAXW, HOLE, np.int8)
        row[:self._width] = FLOOR
        return row

    def _set_width(self, new_w):
        self._left -= (new_w - self._width) / 2   # keep it centered
        self._width = new_w

    def _reach(self):
        """Lanes the ball can surely cross since the last obstacle, even at top speed."""
        frames = (self._since_obstacle + 1) / MAX_SPEED
        return max(1, int(0.1 * (frames - 7)))

    def _plan(self, i):
        """Queue the next row, or a whole feature (ramp, drop or moving blocks)."""
        d = min(1.0, i / HARD_AT)  # difficulty, 0..1
        self._next_feature -= 1
        if (i >= FEATURES_FROM and self._next_feature <= 0 and self._hole_rows == 0
                and self._quiet == 0 and self._since_obstacle >= 8):
            self._next_feature = int(self.rng.integers(int(60 - 25 * d), int(100 - 35 * d)))
            kind = self.rng.choice(["jump", "drop", "mover"], p=[0.35, 0.35, 0.3])
            getattr(self, f"_plan_{kind}")(d)
        else:
            self._plan_row(d)

    def _plan_row(self, d):
        rng = self.rng
        # Every so often pick a new curve and maybe change the width
        # (but not halfway through a two-row hole, which could close its opening).
        self._section -= 1
        if self._section <= 0 and self._hole_rows == 0:
            self._section = int(rng.integers(15, 40))
            self._drift = rng.uniform(-1, 1) * (0.03 + 0.1 * d)
            self._set_width(int(np.clip(self._width + rng.integers(-1, 2), round(5 - 2 * d), 7)))

        row = self._floor()
        obstacle = False
        if self._hole_rows > 0:          # a hole that is two rows long
            row[self._hole_lanes] = HOLE
            self._hole_rows -= 1
        elif self._quiet > 0:
            self._quiet -= 1
        else:
            self._next_obstacle -= 1
            if self._next_obstacle <= 0 and self._width >= 3:
                obstacle = True
                self._next_obstacle = int(rng.integers(int(14 - 6 * d), int(24 - 10 * d) + 1))
                # Always leave two neighbouring lanes open, close enough to the last
                # opening that the ball can get there in time.
                anchor = self._prev_free if self._prev_free is not None else (self._width - 2) // 2
                reach = self._reach()
                free = int(np.clip(anchor + rng.integers(-reach, reach + 1), 0, self._width - 2))
                self._prev_free = free
                others = [j for j in range(self._width) if j not in (free, free + 1)]
                k = int(rng.integers(1, min(len(others), 1 + int(3 * d)) + 1))
                lanes = rng.choice(others, k, replace=False)
                if rng.random() < 0.3:
                    row[lanes] = HOLE
                    self._hole_rows, self._hole_lanes = int(rng.integers(0, 2)), lanes
                else:
                    row[lanes] = BLOCK
        self._emit(row)
        if obstacle:
            self._since_obstacle = 0

    def _plan_drop(self, d):
        """The track steps down: the ball flies off the edge and lands lower down,
        on a platform that may be shifted sideways."""
        rng = self.rng
        self._h -= rng.uniform(1.0, 1.5 + d)
        self._left += rng.uniform(-1, 1) * (0.3 + 0.7 * d)
        self._emit(self._floor())
        self._after_landing()

    def _after_landing(self):
        # The ball can land anywhere across the track, so leave it enough clear
        # track to reach any lane before the next obstacle.
        self._quiet, self._prev_free, self._since_obstacle = 30, None, 0

    def _plan_jump(self, d):
        """A ramp launches the ball over a gap onto a lower platform. With the
        fixed fall curve the ball always flies at least 9 tiles, and the gap is
        at most 6, so the jump can always be made."""
        rng = self.rng
        base, rise = self._h, rng.uniform(1.0, 1.4)
        for k in range(RAMP_ROWS):
            self._emit(self._floor(), base + rise * k / RAMP_ROWS,
                       base + rise * (k + 1) / RAMP_ROWS)
        self._h = base - rng.uniform(0.0, 0.5 + d)
        for _ in range(int(rng.integers(3, 7))):
            self._emit(np.full(MAXW, HOLE, np.int8), gap=True)
        self._left += rng.uniform(-1, 1) * (0.3 + 0.7 * d)
        self._after_landing()

    def _plan_mover(self, d):
        """Rows with a block sliding from side to side. It never reaches two
        neighbouring lanes, and where it can, it keeps clear of the lanes that
        were open at the last obstacle, so there's always a safe way past."""
        rng = self.rng
        if self._width < 4:
            self._set_width(4)
        w = self._width
        p = self._prev_free if self._prev_free is not None else (w - 2) // 2
        p = min(p, w - 2)
        for _ in range(1 + int(rng.random() < 0.6 * d)):
            room_right, room_left = w - 3 - p, p - 1   # how far it can sweep without reaching p
            if max(room_right, room_left) < 1:         # 4 wide, open lanes in the middle:
                room_right, p = 1, 0                   # sweep the right two, ball moves one lane
            if room_right >= 1 and (room_left < 1 or rng.random() < 0.5):
                span = int(rng.integers(1, room_right + 1))   # it sweeps span + 1 lanes
                a = w - 1 - span
            else:
                span = int(rng.integers(1, room_left + 1))
                a = 0
            mover = (a + (span + 1) / 2, span / 2,
                     rng.uniform(0.03, 0.045 + 0.03 * d), rng.uniform(0, 2 * np.pi))
            self._emit(self._floor(), mover=mover)
            self._since_obstacle = 0
            for _ in range(6):
                self._emit(self._floor())
        self._prev_free = p

    # -- looking things up ------------------------------------------------------------
    def _rows(self, zs):
        zs = np.maximum(zs, 0.0)
        rows = zs.astype(int)
        if rows.size:
            self.ensure(int(rows.max()) + 2)
        return zs, rows

    def left_at(self, zs):
        """x of the track's left edge at depth zs."""
        zs, rows = self._rows(zs)
        return self.left0[rows] + (self.left1[rows] - self.left0[rows]) * (zs - rows)

    def height(self, zs):
        """Height of the track surface at depth zs (also defined over holes and gaps)."""
        zs, rows = self._rows(zs)
        return self.h0[rows] + (self.h1[rows] - self.h0[rows]) * (zs - rows)

    def _look(self, xs, zs, t):
        zs, rows = self._rows(zs)
        left = self.left_at(zs)
        j = np.floor(xs - left).astype(int)
        inside = (j >= 0) & (j < MAXW)
        tile = np.where(inside, self.tiles[rows, np.clip(j, 0, MAXW - 1)], HOLE)
        moving = self.mover[rows, 2] > 0
        if moving.any():
            m = self.mover[rows]
            bx = left + m[..., 0] + m[..., 1] * np.sin(m[..., 2] * t + m[..., 3])
            tile = np.where(moving & (np.abs(xs - bx) < 0.5) & (tile == FLOOR), BLOCK, tile)
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

    N_OBS = len(RAY_ANGLES) + 2

    def __init__(self, n, seed=None):
        self.n = n
        self.track = Track(seed)
        self.x = np.zeros(n)
        self.vx = np.zeros(n)
        self.z = np.full(n, 1.0)
        self.y = np.zeros(n)                     # height of the bottom of the ball
        self.vy = np.zeros(n)                    # climb per tile travelled
        self.air = np.zeros(n, dtype=bool)       # flying (off a ramp or a drop)?
        self.speed = START_SPEED
        self.alive = np.ones(n, dtype=bool)
        self.scores = np.zeros(n, dtype=int)     # frames survived (fitness)
        self.death = np.zeros(n, dtype=np.int8)  # 0 alive, 1 fell, 2 hit a block
        self.frames = 0
        self.obs = np.zeros((n, self.N_OBS), np.float32)
        self.observe(np.arange(n))

    @property
    def display_scores(self):
        return self.z.astype(int)

    def step(self, actions):
        idx = np.flatnonzero(self.alive)
        dz = self.speed
        push = (actions[idx] == RIGHT).astype(float) - (actions[idx] == LEFT)
        self.vx[idx] = (self.vx[idx] + push * STEER) * FRICTION
        self.x[idx] += self.vx[idx]
        self.z[idx] += dz
        self.speed = min(MAX_SPEED, self.speed + SPEED_UP)
        self.frames += 1
        self.scores[idx] += 1

        track, t = self.track, self.frames
        x, z = self.x[idx], self.z[idx]
        under = track.at(x, z, t)
        surf = track.height(z)
        # Fly along a curve; if that would go into the floor, roll on the floor instead.
        y = self.y[idx] + self.vy[idx] * dz - 0.5 * FALL_CURVE * dz * dz
        land = (under != HOLE) & (y <= surf)
        rows = z.astype(int)
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
        """Update what balls idx see: 7 ray distances, sideways velocity, speed.
        Only living balls are updated, which keeps late generations fast."""
        self.obs[idx, :len(RAY_ANGLES)] = self.ray_distances(idx) / RAY_MAX
        self.obs[idx, -2] = self.vx[idx] / 0.12
        self.obs[idx, -1] = self.speed / MAX_SPEED


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
RAMP_EDGE = (190, 255, 90)
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
            sunk = [b[3] < track.height(b[2]) - 0.05 for b in balls]
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
        pg, p, screen = self.pg, self.project, self.screen
        tiles = track.tiles[r]
        ln, lf = float(track.left_at(z0)), float(track.left1[r])    # left edge, near/far end
        hn, hf = float(track.height(z0)), float(track.h1[r])        # height, near/far end
        near_pts = [p(ln + k, z0, hn) for k in range(MAXW + 1)]
        far_pts = [p(lf + k, z1, hf) for k in range(MAXW + 1)]
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
                pg.draw.polygon(screen, side, [p(xf, z1, hf), p(xn, z0, hn),
                                               p(xn, z0, hn - T), p(xf, z1, hf - T)])
            if (j == MAXW - 1 or tiles[j + 1] == HOLE) and xn + 1 < self.cam_x:
                pg.draw.polygon(screen, side, [p(xf + 1, z1, hf), p(xn + 1, z0, hn),
                                               p(xn + 1, z0, hn - T), p(xf + 1, z1, hf - T)])
            if front_open[j]:
                pg.draw.polygon(screen, side, [p(xn, z0, hn), p(xn + 1, z0, hn),
                                               p(xn + 1, z0, hn - T), p(xn, z0, hn - T)])
            quad = [far_pts[j], far_pts[j + 1], near_pts[j + 1], near_pts[j]]
            pg.draw.polygon(screen, top, quad)
            pg.draw.polygon(screen, edge, quad, 1)

        h_mid = (float(track.h0[r]) + hf) / 2
        for j in np.flatnonzero(tiles == BLOCK):
            self.draw_block((track.left0[r] + track.left1[r]) / 2 + j, r, h_mid, fog)
        c, amp, w, phase = track.mover[r]
        if w > 0:
            bx = float(track.left_at(r + 0.5)) + c + amp * np.sin(w * t + phase)
            self.draw_block(bx - 0.5, r, h_mid, fog)

    def draw_block(self, x, r, h, fog):
        pg, p = self.pg, self.project
        x0, x1 = x + BLOCK_INSET, x + 1 - BLOCK_INSET
        zb0, zb1, top_y = r + BLOCK_INSET, r + 1 - BLOCK_INSET, h + BLOCK_H
        if zb0 <= self.cam_z + NEAR:
            return
        top = [p(x0, zb1, top_y), p(x1, zb1, top_y), p(x1, zb0, top_y), p(x0, zb0, top_y)]
        pg.draw.polygon(self.screen, lerp(BLOCK_TOP, VOID, fog), top)
        side_x = x1 if x1 < self.cam_x else x0 if x0 > self.cam_x else None
        if side_x is not None:
            side = [p(side_x, zb1, top_y), p(side_x, zb0, top_y), p(side_x, zb0, h),
                    p(side_x, zb1, h)]
            pg.draw.polygon(self.screen, lerp(BLOCK_SIDE, VOID, fog), side)
        front = [p(x0, zb0, top_y), p(x1, zb0, top_y), p(x1, zb0, h), p(x0, zb0, h)]
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

        surf = float(batch.track.height(z))
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
        pg = self.pg
        dists = batch.obs[i, :len(RAY_ANGLES)] * RAY_MAX
        x, z = batch.x[i], batch.z[i]
        start = self.project(x, z, batch.y[i] + 0.05)
        for ang, d in zip(RAY_ANGLES, dists):
            ex, ez = x + np.sin(ang) * d, z + np.cos(ang) * d
            end = self.project(ex, ez, float(batch.track.height(ez)) + 0.05)
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
            if game.display_scores[0] // 100 > before // 100:
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
