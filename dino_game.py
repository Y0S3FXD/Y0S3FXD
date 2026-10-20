"""A tiny Chrome-style dinosaur game.

The game logic (DinoGame) has no graphics dependency, so an AI can play
thousands of games per second without opening a window. Rendering with
pygame is only used when a human plays or when you watch the AI.

Run `python dino_game.py` to play it yourself:
    SPACE / UP  = jump
    DOWN        = duck (also makes you fall faster mid-air)
    R           = restart after game over
"""

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

START_SPEED = 6.0
MAX_SPEED = 14.0
SPEED_UP = 0.002        # speed added every frame

# Actions
NOOP, JUMP, DUCK = 0, 1, 2
N_ACTIONS = 3


class Obstacle:
    def __init__(self, x, w, h, bottom, kind):
        self.x = x              # left edge
        self.w = w
        self.h = h
        self.bottom = bottom    # height above ground of the obstacle's bottom
        self.kind = kind        # "cactus" or "bird"

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
        self.speed = START_SPEED
        self.score = 0
        self.done = False
        self.obstacles = []
        self.next_spawn = 400.0  # pixels until the next obstacle appears
        return self.observe()

    # -- rules ------------------------------------------------------------------
    def _spawn(self):
        # Birds only show up once the game has gotten going.
        if self.score > 600 and self.rng.random() < 0.3:
            bottom = self.rng.choice([5, 34, 60])  # low: jump, mid: duck, high: ignore
            self.obstacles.append(Obstacle(WIDTH, 46, 30, bottom, "bird"))
        else:
            count = self.rng.choice([1, 1, 2, 3])
            w = 17 * count
            h = self.rng.choice([35, 50])
            self.obstacles.append(Obstacle(WIDTH, w, h, 0, "cactus"))
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
        if action == JUMP and on_ground:
            self.vy = JUMP_VELOCITY
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


# --- Drawing (pygame) ------------------------------------------------------------
def draw(screen, font, game, label=""):
    import pygame

    screen.fill((247, 247, 247))
    pygame.draw.line(screen, (83, 83, 83), (0, GROUND_Y), (WIDTH, GROUND_Y), 2)

    x, bottom, w, h = game._dino_box()
    pygame.draw.rect(screen, (83, 83, 83), (x, GROUND_Y - bottom - h, w, h))
    for o in game.obstacles:
        color = (60, 140, 60) if o.kind == "cactus" else (120, 80, 160)
        pygame.draw.rect(screen, color, (o.x, GROUND_Y - o.top, o.w, o.h))

    text = f"{label}  score {game.score}  speed {game.speed:.1f}".strip()
    screen.blit(font.render(text, True, (83, 83, 83)), (10, 10))
    if game.done:
        screen.blit(font.render("GAME OVER - press R", True, (200, 50, 50)),
                    (WIDTH // 2 - 90, 60))
    pygame.display.flip()


def play_human():
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Dino")
    font = pygame.font.SysFont(None, 26)
    clock = pygame.time.Clock()
    game = DinoGame()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r and game.done:
                game.reset()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE] or keys[pygame.K_UP]:
            action = JUMP
        elif keys[pygame.K_DOWN]:
            action = DUCK
        else:
            action = NOOP

        game.step(action)
        draw(screen, font, game, "YOU")
        clock.tick(60)


if __name__ == "__main__":
    play_human()
