import random
import sys
import time

import pygame

# Initialize pygame
pygame.init()

# Constants
GRID_SIZE = 10
CELL_SIZE = 60
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
FPS = 15
PLAYER_LIVES = 3
MAX_SHOTS = 3
SHOT_COOLDOWN = 2  # seconds

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 102, 204)  # Player
GREEN = (0, 255, 0)  # Friendly projectile
RED = (255, 0, 0)  # Enemy or hostile projectile
YELLOW = (255, 255, 0)  # Power-up

# Set up display
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Grid Shooter")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 24)

# Game state
player_pos = [1, 1]
player_dir = [0, -1]  # Facing up
enemies = [[8, 8]]
projectiles = []  # [x, y, (dx, dy), is_friendly]
powerups = [[4, 4]]
player_lives = PLAYER_LIVES
shots_left = MAX_SHOTS
last_shot_time = time.time()

# === Functions ===


def draw_grid():
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLACK, rect, 1)


def draw():
    screen.fill(BLACK)
    draw_grid()

    # Player
    px, py = player_pos
    pygame.draw.rect(
        screen, BLUE, (px * CELL_SIZE, py * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    )

    # Enemies
    for ex, ey in enemies:
        pygame.draw.rect(
            screen, RED, (ex * CELL_SIZE, ey * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        )

    # Powerups
    for pu_x, pu_y in powerups:
        pygame.draw.rect(
            screen, YELLOW, (pu_x * CELL_SIZE, pu_y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        )

    # Projectiles
    for x, y, _, is_friendly in projectiles:
        color = GREEN if is_friendly else RED
        pygame.draw.circle(
            screen,
            color,
            (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2),
            6,
        )

    # Leben anzeigen
    lives_text = font.render(f"Leben: {player_lives}", True, WHITE)
    screen.blit(lives_text, (10, 10))

    pygame.display.flip()


def move_player(dx, dy):
    global player_dir
    new_x = player_pos[0] + dx
    new_y = player_pos[1] + dy
    if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE:
        player_pos[0] = new_x
        player_pos[1] = new_y
        player_dir = [dx, dy]


def shoot():
    global shots_left, last_shot_time
    if shots_left > 0:
        dx, dy = player_dir
        if dx == 0 and dy == 0:
            return
        projectile_pos = [player_pos[0], player_pos[1]]
        projectiles.append([projectile_pos[0], projectile_pos[1], (dx, dy), True])
        shots_left -= 1
        last_shot_time = time.time()


def shoot_enemy():
    global enemies, projectiles
    for ex, ey in enemies:
        dir = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
        projectiles.append([ex, ey, dir, False])


def update_projectiles():
    global projectiles, enemies, player_lives
    new_projectiles = []
    for x, y, (dx, dy), is_friendly in projectiles:
        x += dx
        y += dy
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            if is_friendly:
                if [x, y] in enemies:
                    enemies.remove([x, y])
                else:
                    new_projectiles.append([x, y, (dx, dy), is_friendly])
            else:
                if [x, y] == player_pos:
                    player_lives -= 1
                    print(f"Getroffen! Leben übrig: {player_lives}")
                    if player_lives <= 0:
                        print("Game Over!")
                        pygame.quit()
                        sys.exit()
                else:
                    new_projectiles.append([x, y, (dx, dy), is_friendly])
    projectiles = new_projectiles


def move_enemies():
    for i in range(len(enemies)):
        dx, dy = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
        ex, ey = enemies[i]
        new_x = ex + dx
        new_y = ey + dy
        if (
            0 <= new_x < GRID_SIZE
            and 0 <= new_y < GRID_SIZE
            and [new_x, new_y] != player_pos
        ):
            enemies[i] = [new_x, new_y]


def check_powerups():
    global player_pos, powerups, player_lives
    if player_pos in powerups:
        powerups.remove(player_pos)
        player_lives += 1
        print("Power-up eingesammelt! Leben:", player_lives)


def reset_shots_if_needed():
    global shots_left, last_shot_time
    if shots_left == 0 and time.time() - last_shot_time >= SHOT_COOLDOWN:
        shots_left = MAX_SHOTS


# === Main Game Loop ===

running = True
while running:
    clock.tick(FPS)

    draw()
    update_projectiles()
    move_enemies()
    shoot_enemy()
    check_powerups()
    reset_shots_if_needed()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                move_player(0, -1)
            elif event.key == pygame.K_DOWN:
                move_player(0, 1)
            elif event.key == pygame.K_LEFT:
                move_player(-1, 0)
            elif event.key == pygame.K_RIGHT:
                move_player(1, 0)
            elif event.key == pygame.K_SPACE:
                shoot()

pygame.quit()
sys.exit()
