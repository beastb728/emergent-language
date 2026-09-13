import random

import pygame

from environment.world import World
from environment.agent import Agent


SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

WORLD_WIDTH = 10.0
WORLD_HEIGHT = 10.0

MAX_STEPS = 100
TARGET_RADIUS = 0.35


def world_to_screen(position):
    x = int(position[0] / WORLD_WIDTH * SCREEN_WIDTH)
    y = int(position[1] / WORLD_HEIGHT * SCREEN_HEIGHT)

    return x, y


def distance(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]

    return (dx * dx + dy * dy) ** 0.5


pygame.init()

screen = pygame.display.set_mode(
    (SCREEN_WIDTH, SCREEN_HEIGHT)
)

pygame.display.set_caption("Emergent Language — V1")

clock = pygame.time.Clock()

font = pygame.font.Font(None, 28)

world = World()

agent_a = Agent(world.agent_a)
agent_b = Agent(world.agent_b)

episode = 1
step = 0
successes = 0

running = True

while running:

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_r:
                world.reset()

                agent_a = Agent(world.agent_a)
                agent_b = Agent(world.agent_b)

                step = 0
                episode += 1

    # -------------------------
    # RANDOM AGENT B
    # -------------------------

    action = random.randint(0, 4)

    agent_b.move(action)

    step += 1

    # -------------------------
    # CHECK SUCCESS
    # -------------------------

    target_distance = distance(
        agent_b.position,
        world.target
    )

    success = target_distance < TARGET_RADIUS

    if success:
        successes += 1

        print(
            f"Episode {episode}: SUCCESS "
            f"in {step} steps"
        )

        world.reset()

        agent_a = Agent(world.agent_a)
        agent_b = Agent(world.agent_b)

        episode += 1
        step = 0

    elif step >= MAX_STEPS:

        print(
            f"Episode {episode}: FAILED"
        )

        world.reset()

        agent_a = Agent(world.agent_a)
        agent_b = Agent(world.agent_b)

        episode += 1
        step = 0

    # -------------------------
    # RENDER
    # -------------------------

    screen.fill((30, 30, 30))

    # Agent A
    pygame.draw.circle(
        screen,
        (80, 160, 255),
        world_to_screen(agent_a.position),
        15
    )

    # Agent B
    pygame.draw.circle(
        screen,
        (255, 100, 100),
        world_to_screen(agent_b.position),
        15
    )

    # Target
    pygame.draw.circle(
        screen,
        (255, 220, 80),
        world_to_screen(world.target),
        10
    )

    # HUD
    success_rate = (
        successes / max(episode - 1, 1)
    ) * 100

    text = font.render(
        f"Episode: {episode}   "
        f"Step: {step}/{MAX_STEPS}   "
        f"Success: {success_rate:.1f}%",
        True,
        (230, 230, 230)
    )

    screen.blit(text, (20, 20))

    pygame.display.flip()

    clock.tick(20)


pygame.quit()