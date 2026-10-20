"""A Slope-style game: a ball rolls down an endless neon track that keeps
getting faster, curvier and narrower. Steer around the red blocks, and don't
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

# --- World (units: 1 = one tile; x = sideways, z = down the slope) ---------------
MAXW = 9                # widest a track row can be, in tiles
BALL_R = 0.25
HIT_R = 0.17            # a block this close to the ball's center counts as a hit
BLOCK_INSET = 0.08      # blocks are drawn this much smaller than their tile

START_SPEED = 0.15      # tiles per frame
MAX_SPEED = 0.38
SPEED_UP = 0.00004
STEER = 0.014           # sideways push per frame
FRICTION = 0.88         # share of sideways velocity kept each frame (slidey steering)
HARD_AT = 1500          # rows until the track reaches full difficulty

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

    Row i covers z in [i, i+1). Its left edge slides from left0[i] to left1[i]
    along the row, which is what makes the track curve smoothly.
    """

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.left0 = np.zeros(0)
        self.left1 = np.zeros(0)
        self.tiles = np.zeros((0, MAXW), np.int8)     # FLOOR / HOLE / BLOCK
        self.n = 0
        self._left, self._width, self._drift = -2.5, 5, 0.0
        self._section = 30          # rows until the curve/width changes
        self._next_obstacle = 40    # rows until the next obstacle row
        self._hole_rows, self._hole_lanes = 0, None

    def ensure(self, rows):
        """Make sure rows 0..rows-1 exist (built in chunks to avoid constant copying)."""
        if rows <= self.n:
            return
        new = [self._make_row(i) for i in range(self.n, rows + 64)]
        self.n = rows + 64
        self.left0 = np.concatenate([self.left0, [r[0] for r in new]])
        self.left1 = np.concatenate([self.left1, [r[1] for r in new]])
        self.tiles = np.concatenate([self.tiles, np.array([r[2] for r in new])])

    def _make_row(self, i):
        rng = self.rng
        d = min(1.0, i / HARD_AT)  # difficulty, 0..1

        # Every so often pick a new curve and maybe change the width.
        self._section -= 1
        if self._section <= 0:
            self._section = int(rng.integers(15, 40))
            self._drift = rng.uniform(-1, 1) * (0.03 + 0.1 * d)
            new_w = int(np.clip(self._width + rng.integers(-1, 2), round(5 - 2 * d), 7))
            self._left -= (new_w - self._width) / 2   # keep it centered
            self._width = new_w
        left0 = self._left
        self._left += self._drift

        row = np.full(MAXW, HOLE, np.int8)
        row[:self._width] = FLOOR
        if self._hole_rows > 0:          # a hole that is two rows long
            row[self._hole_lanes] = HOLE
            self._hole_rows -= 1
            return left0, self._left, row

        self._next_obstacle -= 1
        if self._next_obstacle <= 0 and self._width >= 3:
            self._next_obstacle = int(rng.integers(int(14 - 6 * d), int(24 - 10 * d) + 1))
            # Always leave two neighbouring lanes open so there's a way through.
            free = int(rng.integers(0, self._width - 1))
            others = [j for j in range(self._width) if j not in (free, free + 1)]
            k = int(rng.integers(1, min(len(others), 1 + int(3 * d)) + 1))
            lanes = rng.choice(others, k, replace=False)
            if rng.random() < 0.3:
                row[lanes] = HOLE
                self._hole_rows, self._hole_lanes = int(rng.integers(0, 2)), lanes
            else:
                row[lanes] = BLOCK
        return left0, self._left, row

    def left_at(self, rows, zs):
        """x of the track's left edge at depth zs (rows = the row each z is in)."""
        return self.left0[rows] + (self.left1[rows] - self.left0[rows]) * (zs - rows)

    def at(self, xs, zs):
        """Tile type at world points (arrays of any shape). Off the track is HOLE."""
        zs = np.maximum(zs, 0.0)
        if zs.size == 0:
            return np.zeros(zs.shape, np.int8)
        rows = zs.astype(int)
        self.ensure(int(rows.max()) + 2)
        j = np.floor(xs - self.left_at(rows, zs)).astype(int)
        inside = (j >= 0) & (j < MAXW)
        return np.where(inside, self.tiles[rows, np.clip(j, 0, MAXW - 1)], HOLE)


class SlopeBatch:
    """n balls on the same track, stepped together. n=1 for a human player."""

    N_OBS = len(RAY_ANGLES) + 2

    def __init__(self, n, seed=None):
        self.n = n
        self.track = Track(seed)
        self.x = np.zeros(n)
        self.vx = np.zeros(n)
        self.z = np.full(n, 1.0)
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
        a = self.alive
        push = (actions == RIGHT).astype(float) - (actions == LEFT)
        self.vx[a] = (self.vx[a] + push[a] * STEER) * FRICTION
        self.x[a] += self.vx[a]
        self.z[a] += self.speed
        self.speed = min(MAX_SPEED, self.speed + SPEED_UP)
        self.frames += 1
        self.scores[a] += 1

        idx = np.flatnonzero(a)
        x, z = self.x[idx], self.z[idx]
        at = self.track.at
        under = at(x, z)
        fell = under == HOLE
        crashed = ((under == BLOCK) | (at(x - HIT_R, z) == BLOCK) |
                   (at(x + HIT_R, z) == BLOCK) | (at(x, z + HIT_R) == BLOCK))
        self.death[idx[fell]] = 1
        self.death[idx[crashed]] = 2
        self.alive[idx[fell | crashed]] = False
        self.observe(np.flatnonzero(self.alive))

    def ray_distances(self, idx):
        """(len(idx), 7): how far each ray gets before hitting an edge, hole or block."""
        xs = self.x[idx, None, None] + np.sin(RAY_ANGLES)[None, :, None] * RAY_T
        zs = self.z[idx, None, None] + np.cos(RAY_ANGLES)[None, :, None] * RAY_T
        bad = self.track.at(xs, zs) != FLOOR
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
CAM_H = 1.5             # camera height above the track
CAM_BACK = 3.0          # camera distance behind the ball
NEAR = 0.6              # nothing closer than this to the camera is drawn
VIEW_ROWS = 45
THICKNESS = 0.35        # how thick the track tiles are

SKY_TOP, SKY_BOTTOM, VOID = (4, 4, 18), (34, 12, 66), (2, 2, 10)
TILE, TILE_EDGE, TILE_SIDE = (8, 40, 22), (0, 255, 120), (0, 90, 50)
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
        self.cam_x = 0.0
        self.cam_z = 0.0
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
        return (WIDTH / 2 + (x - self.cam_x) * s, VANISH_Y + (CAM_H - y) * s)

    def update_population(self, batch, lead, frames=1):
        if batch.alive[lead]:
            # The camera glides after the ball sideways, like the real game.
            self.cam_x += (batch.x[lead] - self.cam_x) * min(1.0, 0.12 * frames)
            self.fall_t = 0
        else:
            self.fall_t += frames

    def ball_pos(self, batch, i):
        """Where to draw ball i as (x, z, y). A ball that fell keeps rolling as it drops."""
        x, z = batch.x[i], batch.z[i]
        if batch.alive[i] or batch.death[i] != 1:
            return x, z, BALL_R
        t = self.fall_t
        return x + batch.vx[i] * t, z + batch.speed * t, BALL_R - 0.012 * t * t

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
            for b in balls:
                if b[3] < 0:  # falling balls go under the tiles
                    self.draw_ball(batch, *b, b[0] == lead)
            self.draw_row(track, r, z0, z1, fog)
            for b in balls:
                if b[3] >= 0:
                    self.draw_ball(batch, *b, b[0] == lead)

        if self.show_rays and batch.alive[lead]:
            self.draw_rays(batch, lead)
        self.draw_score(batch, lead, hi_score)

    def draw_row(self, track, r, z0, z1, fog):
        pg, p, screen = self.pg, self.project, self.screen
        tiles = track.tiles[r]
        in_front = track.tiles[r - 1] if r > 0 else tiles   # the row nearer to us
        ln = float(track.left_at(r, z0))     # left edge at the near end of the row
        lf = float(track.left1[r])           # left edge at the far end
        near_pts = [p(ln + k, z0) for k in range(MAXW + 1)]
        far_pts = [p(lf + k, z1) for k in range(MAXW + 1)]
        top, edge = lerp(TILE, VOID, fog), lerp(TILE_EDGE, SKY_BOTTOM, fog)
        side, T = lerp(TILE_SIDE, VOID, fog), -THICKNESS

        for j in np.flatnonzero(tiles != HOLE):
            # The tile's thickness shows where it borders a hole or the track edge.
            xn, xf = ln + j, lf + j
            if (j == 0 or tiles[j - 1] == HOLE) and xn > self.cam_x:
                pg.draw.polygon(screen, side, [p(xf, z1), p(xn, z0), p(xn, z0, T), p(xf, z1, T)])
            if (j == MAXW - 1 or tiles[j + 1] == HOLE) and xn + 1 < self.cam_x:
                pg.draw.polygon(screen, side, [p(xf + 1, z1), p(xn + 1, z0),
                                               p(xn + 1, z0, T), p(xf + 1, z1, T)])
            if z0 == r and in_front[j] == HOLE:
                pg.draw.polygon(screen, side, [p(xn, z0), p(xn + 1, z0),
                                               p(xn + 1, z0, T), p(xn, z0, T)])
            quad = [far_pts[j], far_pts[j + 1], near_pts[j + 1], near_pts[j]]
            pg.draw.polygon(screen, top, quad)
            pg.draw.polygon(screen, edge, quad, 1)

        for j in np.flatnonzero(tiles == BLOCK):
            self.draw_block((track.left0[r] + track.left1[r]) / 2 + j, r, fog)

    def draw_block(self, x, r, fog):
        pg, p = self.pg, self.project
        x0, x1 = x + BLOCK_INSET, x + 1 - BLOCK_INSET
        zb0, zb1, h = r + BLOCK_INSET, r + 1 - BLOCK_INSET, 0.75
        if zb0 <= self.cam_z + NEAR:
            return
        top = [p(x0, zb1, h), p(x1, zb1, h), p(x1, zb0, h), p(x0, zb0, h)]
        pg.draw.polygon(self.screen, lerp(BLOCK_TOP, VOID, fog), top)
        side_x = x1 if x1 < self.cam_x else x0 if x0 > self.cam_x else None
        if side_x is not None:
            side = [p(side_x, zb1, h), p(side_x, zb0, h), p(side_x, zb0), p(side_x, zb1)]
            pg.draw.polygon(self.screen, lerp(BLOCK_SIDE, VOID, fog), side)
        front = [p(x0, zb0, h), p(x1, zb0, h), p(x1, zb0), p(x0, zb0)]
        pg.draw.polygon(self.screen, lerp(BLOCK_FRONT, VOID, fog), front)
        pg.draw.polygon(self.screen, lerp(BLOCK_TOP, VOID, fog), front, 1)

    def draw_ball(self, batch, i, x, z, y, is_lead):
        pg, screen = self.pg, self.screen
        dz = z - self.cam_z
        if dz < 0.8:
            return
        cx, cy = self.project(x, z, y)
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
        start = self.project(x, z, 0.05)
        for ang, d in zip(RAY_ANGLES, dists):
            end = self.project(x + np.sin(ang) * d, z + np.cos(ang) * d, 0.05)
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
            before = game.display_scores[0]
            game.step(np.array([action]))
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
