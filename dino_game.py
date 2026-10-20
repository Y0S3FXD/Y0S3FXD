"""A Chrome-style dinosaur game.

The game logic (DinoGame) has no graphics dependency, so an AI can play
thousands of games per second without opening a window. Rendering with
pygame is only used when a human plays or when you watch the AI.

Run `python dino_game.py` to play it yourself:
    SPACE / UP  = jump (tap for a small hop, hold to jump higher)
    DOWN        = duck (also makes you fall faster mid-air)
    ESC         = quit
"""

import os
import random

import numpy as np

# --- World constants (y is height above the ground, in pixels) ---------------
WIDTH, HEIGHT = 800, 250
GROUND_Y = 200          # screen y of the ground line (only used for drawing)

DINO_X = 60
DINO_W = 40
DINO_H = 44
DINO_DUCK_H = 26

GRAVITY = 0.9
FAST_FALL_GRAVITY = 2.5
JUMP_VELOCITY = 14.0
SHORT_JUMP_VELOCITY = 5.0   # letting go of jump early cuts the rise to this
MIN_JUMP_HEIGHT = 30        # ...but only after reaching this height

START_SPEED = 6.0
MAX_SPEED = 14.0
SPEED_UP = 0.002        # speed added every frame

BIRDS_AFTER = 600       # frames before birds can appear

# Actions
NOOP, JUMP, DUCK = 0, 1, 2
N_ACTIONS = 3


class Obstacle:
    def __init__(self, x, w, h, bottom, kind, count=1):
        self.x = x              # left edge
        self.w = w
        self.h = h
        self.bottom = bottom    # height above ground of the obstacle's bottom
        self.kind = kind        # "cactus" or "bird"
        self.count = count      # cacti in a group

    @property
    def top(self):
        return self.bottom + self.h


class DinoGame:
    """Game state + rules. Call reset(), then step(action) once per frame."""

    N_OBS = 8  # size of the observation vector (see observe())

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.y = 0.0
        self.vy = 0.0
        self.ducking = False
        self.just_jumped = False
        self.speed = START_SPEED
        self.score = 0          # frames survived (what the AI is rewarded for)
        self.distance = 0.0     # pixels run (what the score counter shows)
        self.done = False
        self.obstacles = []
        self.next_spawn = 400.0  # pixels until the next obstacle appears
        return self.observe()

    @property
    def display_score(self):
        """The number shown in the top right, like the real game."""
        return int(self.distance / 40)

    # -- rules ------------------------------------------------------------------
    def _spawn(self):
        if self.score > BIRDS_AFTER and self.rng.random() < 0.3:
            bottom = self.rng.choice([5, 34, 60])  # low: jump, mid: duck, high: ignore
            self.obstacles.append(Obstacle(WIDTH, 46, 30, bottom, "bird"))
        else:
            count = self.rng.choice([1, 1, 2, 3])
            h = self.rng.choice([35, 50])
            self.obstacles.append(Obstacle(WIDTH, 17 * count, h, 0, "cactus", count))
        # Gap grows with speed so the game stays (barely) beatable.
        self.next_spawn = self.rng.uniform(self.speed * 40, self.speed * 80)

    def _dino_box(self):
        h = DINO_DUCK_H if self.ducking else DINO_H
        return DINO_X, self.y, DINO_W, h  # x, bottom, w, h

    def _hit(self, o):
        x, bottom, w, h = self._dino_box()
        # Shrink the hitboxes a bit, like the real game does, so it feels fair.
        pad = 4
        return (x + pad < o.x + o.w - pad and o.x + pad < x + w - pad and
                bottom + pad < o.top - pad and o.bottom + pad < bottom + h - pad)

    def step(self, action):
        """Advance one frame. Returns (observation, reward, done)."""
        if self.done:
            return self.observe(), 0.0, True

        on_ground = self.y <= 0
        self.just_jumped = action == JUMP and on_ground
        if self.just_jumped:
            self.vy = JUMP_VELOCITY
        elif action != JUMP and self.y > MIN_JUMP_HEIGHT and self.vy > SHORT_JUMP_VELOCITY:
            self.vy = SHORT_JUMP_VELOCITY  # jump released early: short hop
        self.ducking = action == DUCK and on_ground

        gravity = FAST_FALL_GRAVITY if (action == DUCK and not on_ground) else GRAVITY
        self.vy -= gravity
        self.y = max(0.0, self.y + self.vy)
        if self.y == 0:
            self.vy = 0.0

        for o in self.obstacles:
            o.x -= self.speed
        self.obstacles = [o for o in self.obstacles if o.x + o.w > 0]

        self.next_spawn -= self.speed
        if self.next_spawn <= 0:
            self._spawn()

        self.distance += self.speed
        self.speed = min(MAX_SPEED, self.speed + SPEED_UP)
        self.score += 1

        if any(self._hit(o) for o in self.obstacles):
            self.done = True
            return self.observe(), -100.0, True
        return self.observe(), 1.0, False

    # -- what the AI "sees" ---------------------------------------------------------
    def observe(self):
        """8 numbers, all roughly in [0, 1], describing the next two obstacles."""
        ahead = [o for o in self.obstacles if o.x + o.w > DINO_X]
        first = ahead[0] if ahead else None
        second = ahead[1] if len(ahead) > 1 else None
        return np.array([
            (first.x - DINO_X) / WIDTH if first else 1.0,   # distance to next obstacle
            first.w / 60 if first else 0.0,                 # its width
            first.bottom / 100 if first else 0.0,           # its bottom height
            first.top / 100 if first else 0.0,              # its top height
            (second.x - DINO_X) / WIDTH if second else 1.0, # distance to the one after
            self.y / 150,                                   # dino height
            self.vy / JUMP_VELOCITY,                        # dino vertical speed
            self.speed / MAX_SPEED,                         # game speed
        ], dtype=np.float32)


class DinoBatch:
    """n games on the same level, stepped together (what the trainer uses)."""

    N_OBS = DinoGame.N_OBS

    def __init__(self, n, seed=None):
        self.n = n
        self.games = [DinoGame(seed) for _ in range(n)]
        self.alive = np.ones(n, dtype=bool)
        self.scores = np.zeros(n, dtype=int)
        self.obs = np.stack([g.observe() for g in self.games])
        self.frames = 0

    def step(self, actions):
        for i in np.flatnonzero(self.alive):
            obs, _, done = self.games[i].step(actions[i])
            self.obs[i] = obs
            self.scores[i] = self.games[i].score
            if done:
                self.alive[i] = False
        self.frames += 1

    @property
    def display_scores(self):
        return np.array([g.display_score for g in self.games])


# --- Drawing (pygame) ------------------------------------------------------------
DAY_BG, DAY_INK = (247, 247, 247), (83, 83, 83)
NIGHT_BG, NIGHT_INK = (32, 33, 36), (210, 210, 210)
NIGHT_EVERY = 700       # score between nights
NIGHT_LENGTH = 200      # score a night lasts


def lerp(a, b, t):
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


class Renderer:
    """Draws the world: sky, clouds, moon/stars, ground, obstacles, dino, score."""

    def __init__(self, screen, seed=0):
        import pygame
        self.pg = pygame
        self.screen = screen
        self.rng = random.Random(seed)
        self.score_font = pygame.font.SysFont("couriernew,dejavusansmono,monospace", 22,
                                              bold=True)
        self.night = 0.0        # 0 = day, 1 = night, fades in between
        self.ground_offset = 0.0
        self.clouds = [[self.rng.uniform(0, WIDTH), self.rng.uniform(40, 120)]
                       for _ in range(3)]
        self.stars = [(self.rng.uniform(0, WIDTH), self.rng.uniform(15, 130))
                      for _ in range(12)]
        self.moon_x = WIDTH * 0.7
        # Pebbles and dashes under the ground line, scrolled for a sense of speed.
        self.bumps = [(self.rng.uniform(0, WIDTH * 2), self.rng.choice([3, 5, 6]),
                       self.rng.choice([3, 6, 10])) for _ in range(60)]

    @property
    def bg(self):
        return lerp(DAY_BG, NIGHT_BG, self.night)

    @property
    def ink(self):
        return lerp(DAY_INK, NIGHT_INK, self.night)

    def update(self, game, frames=1):
        """Move scenery along. Call once per drawn frame."""
        if game.done:
            return
        s = game.display_score
        is_night = s >= NIGHT_EVERY and s % NIGHT_EVERY < NIGHT_LENGTH
        target = 1.0 if is_night else 0.0
        self.night += max(-0.03 * frames, min(0.03 * frames, target - self.night))

        self.ground_offset = (self.ground_offset + game.speed * frames) % (WIDTH * 2)
        for c in self.clouds:
            c[0] -= game.speed * 0.15 * frames
            if c[0] < -60:
                c[0] = WIDTH + self.rng.uniform(0, 300)
                c[1] = self.rng.uniform(40, 120)
        self.moon_x -= 0.1 * frames
        if self.moon_x < -40:
            self.moon_x = WIDTH + 40

    # -- pieces --------------------------------------------------------------------
    def draw_world(self, game):
        pg, screen = self.pg, self.screen
        screen.fill(self.bg, (0, 0, WIDTH, HEIGHT))
        ink = self.ink

        if self.night > 0.05:
            star = lerp(self.bg, (230, 230, 230), self.night)
            for x, y in self.stars:
                pg.draw.rect(screen, star, (x, y, 2, 2))
            mx, my = int(self.moon_x), 50
            pg.draw.circle(screen, star, (mx, my), 16)
            pg.draw.circle(screen, self.bg, (mx + 8, my - 4), 14)  # crescent

        cloud = lerp((220, 220, 220), (70, 72, 78), self.night)
        for x, y in self.clouds:
            self.draw_cloud(x, y, cloud)

        pg.draw.line(screen, ink, (0, GROUND_Y), (WIDTH, GROUND_Y), 2)
        for bx, w, dy in self.bumps:
            x = (bx - self.ground_offset) % (WIDTH * 2)
            if x < WIDTH:
                pg.draw.rect(screen, ink, (x, GROUND_Y + dy, w, 2))

        for o in game.obstacles:
            if o.kind == "cactus":
                self.draw_cactus(o, ink)
            else:
                self.draw_bird(o, ink, game.score)

    def draw_cloud(self, x, y, color):
        pg = self.pg
        for cx, cy, w, h in ((0, 8, 46, 12), (8, 2, 20, 14), (20, 0, 18, 16)):
            pg.draw.ellipse(self.screen, color, (x + cx, y + cy, w, h))

    def draw_cactus(self, o, color):
        pg = self.pg
        piece = o.w / o.count
        top = GROUND_Y - o.top
        arm_w = 4
        for k in range(o.count):
            left = o.x + k * piece
            trunk_w = piece - 2 * arm_w - 1
            tx = left + arm_w
            pg.draw.rect(self.screen, color, (tx, top, trunk_w, o.h), border_radius=4)
            arm_h = o.h * 0.35
            # left arm, lower
            ay = top + o.h * 0.3
            pg.draw.rect(self.screen, color, (left, ay, arm_w, arm_h), border_radius=2)
            pg.draw.rect(self.screen, color, (left, ay + arm_h - arm_w, arm_w + 1, arm_w))
            # right arm, higher
            rx = tx + trunk_w
            ay = top + o.h * 0.15
            pg.draw.rect(self.screen, color, (rx, ay, arm_w, arm_h), border_radius=2)
            pg.draw.rect(self.screen, color, (rx - 1, ay + arm_h - arm_w, arm_w + 1, arm_w))

    def draw_bird(self, o, color, frame):
        pg = self.pg
        x, top = o.x, GROUND_Y - o.top
        mid = top + o.h / 2
        pg.draw.rect(self.screen, color, (x + 12, mid - 3, 26, 8))             # body
        pg.draw.polygon(self.screen, color, [(x + 12, mid - 3), (x + 2, mid + 1),
                                             (x + 12, mid + 3)])                # beak
        pg.draw.rect(self.screen, color, (x + 10, mid - 7, 8, 6))               # head
        pg.draw.rect(self.screen, color, (x + 38, mid - 2, 8, 3))               # tail
        if (frame // 10) % 2:  # flap
            pg.draw.polygon(self.screen, color, [(x + 18, mid - 3), (x + 30, mid - 3),
                                                 (x + 22, top - 2)])
        else:
            pg.draw.polygon(self.screen, color, [(x + 18, mid + 5), (x + 30, mid + 5),
                                                 (x + 24, top + o.h + 2)])

    def draw_dino(self, game, color=None):
        """Draw a T-rex at the dino's position. color may include alpha for ghosts."""
        pg = self.pg
        color = color or (*self.ink, 255)
        surf = pg.Surface((DINO_W + 20, DINO_H), pg.SRCALPHA)
        eye = (*self.bg, color[3])
        legs_up = game.y == 0 and not game.done and (game.score // 6) % 2  # running
        if game.ducking:
            top = DINO_H - DINO_DUCK_H
            pg.draw.rect(surf, color, (0, top + 4, 44, 14))         # long low body
            pg.draw.rect(surf, color, (36, top, 22, 14))            # head forward
            pg.draw.rect(surf, eye, (40, top + 3, 3, 3))
        else:
            pg.draw.rect(surf, color, (20, 0, 22, 14))              # head
            pg.draw.rect(surf, color, (32, 14, 10, 4))              # jaw
            pg.draw.rect(surf, color, (6, 14, 22, 20))              # body
            pg.draw.rect(surf, color, (0, 16, 8, 8))                # tail
            pg.draw.rect(surf, color, (28, 20, 6, 3))               # arm
            if game.done:  # X eyes and open mouth
                pg.draw.line(surf, eye, (24, 2), (29, 7), 2)
                pg.draw.line(surf, eye, (29, 2), (24, 7), 2)
                pg.draw.rect(surf, eye, (34, 10, 8, 3))
            else:
                pg.draw.rect(surf, eye, (25, 3, 3, 3))
        pg.draw.rect(surf, color, (10, DINO_H - 10, 5, 10 - 4 * legs_up))
        pg.draw.rect(surf, color, (20, DINO_H - 10, 5, 6 + 4 * legs_up))
        self.screen.blit(surf, (DINO_X, GROUND_Y - game.y - DINO_H))

    def draw_score(self, game, hi_score=0):
        s = game.display_score
        # Every 100 points the score blinks for a moment, like the real game.
        flashing = s >= 100 and s % 100 < 6 and not game.done
        shown = s - s % 100 if flashing else s
        text = f"{shown:05d}"
        if flashing and (game.score // 8) % 2:
            text = ""
        if hi_score:
            text = f"HI {hi_score:05d}  " + (text or " " * 5)
        img = self.score_font.render(text, True, self.ink)
        self.screen.blit(img, (WIDTH - img.get_width() - 15, 12))

    def draw_game_over(self):
        pg, ink = self.pg, self.ink
        img = self.score_font.render("G A M E   O V E R", True, ink)
        self.screen.blit(img, (WIDTH // 2 - img.get_width() // 2, 70))
        # restart icon: circular arrow
        cx, cy = WIDTH // 2, 120
        pg.draw.arc(self.screen, ink, (cx - 14, cy - 14, 28, 28), 0.6, 5.8, 4)
        pg.draw.polygon(self.screen, ink, [(cx + 8, cy - 14), (cx + 16, cy - 4),
                                           (cx + 4, cy - 2)])

    def update_population(self, batch, lead, frames=1):
        self.update(batch.games[lead], frames)

    def draw_population(self, batch, lead, hi_score=0):
        """Leader drawn solid, every other living dino as a faint ghost."""
        game = batch.games[lead]
        self.draw_world(game)
        for i in np.flatnonzero(batch.alive):
            if i != lead:
                hue = (i * 47) % 200
                self.draw_dino(batch.games[i], (hue, 120, 255 - hue, 45))
        self.draw_dino(game)
        self.draw_score(game, hi_score)

    def draw(self, game, hi_score=0):
        self.draw_world(game)
        self.draw_dino(game)
        self.draw_score(game, hi_score)
        if game.done:
            self.draw_game_over()


# --- Sound (generated, no files needed) --------------------------------------------
class Sounds:
    def __init__(self):
        import pygame
        self.ok = False
        try:
            pygame.mixer.init(44100, -16, 1)
            freq, _, channels = pygame.mixer.get_init()
            self.jump = self._beep(pygame, freq, channels, [(740, 0.06)])
            self.point = self._beep(pygame, freq, channels, [(1050, 0.08), (1400, 0.12)])
            self.die = self._beep(pygame, freq, channels, [(180, 0.08), (130, 0.18)])
            self.ok = True
        except Exception:
            pass  # no audio device: play silently

    @staticmethod
    def _beep(pygame, rate, channels, notes):
        parts = []
        for hz, secs in notes:
            t = np.arange(int(rate * secs)) / rate
            parts.append(np.sign(np.sin(2 * np.pi * hz * t)) * 0.15)  # square wave
        wave = (np.concatenate(parts) * 32767).astype(np.int16)
        if channels > 1:
            wave = np.repeat(wave[:, None], channels, axis=1)
        return pygame.sndarray.make_sound(np.ascontiguousarray(wave))

    def play(self, name):
        if self.ok:
            getattr(self, name).play()


HI_SCORE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "highscore.txt")


def load_hi_score(path=HI_SCORE_FILE):
    try:
        with open(path) as f:
            return int(f.read().strip() or 0)
    except (OSError, ValueError):
        return 0


def save_hi_score(score, path=HI_SCORE_FILE):
    try:
        with open(path, "w") as f:
            f.write(str(score))
    except OSError:
        pass


def play_human():
    import pygame

    pygame.mixer.pre_init(44100, -16, 1, 512)
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Dino")
    clock = pygame.time.Clock()
    renderer = Renderer(screen)
    sounds = Sounds()
    hint_font = pygame.font.SysFont(None, 26)

    game = DinoGame()
    hi_score = load_hi_score()
    started = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and
                                             event.key == pygame.K_ESCAPE):
                return
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_SPACE, pygame.K_UP):
                if not started:
                    started = True
                elif game.done:
                    game.reset()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE] or keys[pygame.K_UP]:
            action = JUMP
        elif keys[pygame.K_DOWN]:
            action = DUCK
        else:
            action = NOOP

        if started and not game.done:
            before = game.display_score
            game.step(action)
            if game.just_jumped:
                sounds.play("jump")
            if game.display_score // 100 > before // 100:
                sounds.play("point")
            if game.done:
                sounds.play("die")
                if game.display_score > hi_score:
                    hi_score = game.display_score
                    save_hi_score(hi_score)

        renderer.update(game)
        renderer.draw(game, hi_score)
        if not started:
            img = hint_font.render("press SPACE to start", True, renderer.ink)
            screen.blit(img, (WIDTH // 2 - img.get_width() // 2, 90))
        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    play_human()
