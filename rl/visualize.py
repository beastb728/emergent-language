import pygame
import torch

from environment.rl_world import RLWorld
from rl.dqn import DQNAgent


SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

MODEL_PATH = "models/dqn_navigation_final.pt"

FPS = 20


def world_to_screen(position):

    x = int(
        position[0] / 10.0 * SCREEN_WIDTH
    )

    y = int(
        position[1] / 10.0 * SCREEN_HEIGHT
    )

    return x, y


pygame.init()

screen = pygame.display.set_mode(
    (SCREEN_WIDTH, SCREEN_HEIGHT)
)

pygame.display.set_caption(
    "Emergent Language — V1.5"
)

clock = pygame.time.Clock()

font = pygame.font.Font(None, 30)


world = RLWorld()

agent = DQNAgent()

agent.policy_net.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

# No exploration during visualization
agent.epsilon = 0.0


state = world.reset()

episode = 1
successes = 0
paused = False

running = True


while running:

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:

            if event.key == pygame.K_SPACE:
                paused = not paused

            elif event.key == pygame.K_r:

                state = world.reset()
                episode += 1

    if not paused:

        action = agent.choose_action(state)

        next_state, reward, done, reached = (
            world.step(action)
        )

        state = next_state

        if reached:

            successes += 1

            print(
                f"Episode {episode}: "
                f"SUCCESS in "
                f"{world.step_count} steps"
            )

            state = world.reset()
            episode += 1

        elif done:

            print(
                f"Episode {episode}: FAILED"
            )

            state = world.reset()
            episode += 1

    # -------------------------
    # RENDER
    # -------------------------

    screen.fill((30, 30, 30))

    # Target
    pygame.draw.circle(
        screen,
        (255, 220, 80),
        world_to_screen(world.target),
        12
    )

    # Agent
    pygame.draw.circle(
        screen,
        (255, 100, 100),
        world_to_screen(world.agent),
        16
    )

    # Target label
    target_text = font.render(
        "TARGET",
        True,
        (230, 230, 230)
    )

    screen.blit(
        target_text,
        (
            world_to_screen(world.target)[0] + 15,
            world_to_screen(world.target)[1] - 15
        )
    )

    # HUD
    success_rate = (
        successes / max(episode - 1, 1)
    ) * 100

    lines = [
        f"Episode: {episode}",
        f"Step: {world.step_count}/{world.max_steps}",
        f"Success: {success_rate:.1f}%",
        "",
        "SPACE = pause",
        "R = reset",
    ]

    for i, line in enumerate(lines):

        text = font.render(
            line,
            True,
            (230, 230, 230)
        )

        screen.blit(
            text,
            (20, 20 + i * 30)
        )

    if paused:

        paused_text = font.render(
            "PAUSED",
            True,
            (255, 255, 255)
        )

        screen.blit(
            paused_text,
            (SCREEN_WIDTH - 130, 20)
        )

    pygame.display.flip()

    clock.tick(FPS)


pygame.quit()