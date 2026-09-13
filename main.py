import pygame

from environment.world import World
from environment.agent import Agent
from environment.communication import PredefinedProtocol


SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

WORLD_WIDTH = 10.0
WORLD_HEIGHT = 10.0

MAX_STEPS = 100
TARGET_RADIUS = 0.35


def world_to_screen(position):
    x = int(
        position[0] / WORLD_WIDTH * SCREEN_WIDTH
    )

    y = int(
        position[1] / WORLD_HEIGHT * SCREEN_HEIGHT
    )

    return x, y


def distance(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]

    return (dx * dx + dy * dy) ** 0.5


def choose_direction(agent_position, target_position):
    """
    Move toward the target using the largest positional difference.
    """

    dx = target_position[0] - agent_position[0]
    dy = target_position[1] - agent_position[1]

    if abs(dx) > abs(dy):
        if dx > 0:
            return 4  # right
        else:
            return 3  # left

    if dy > 0:
        return 1  # up

    return 2  # down


pygame.init()

screen = pygame.display.set_mode(
    (SCREEN_WIDTH, SCREEN_HEIGHT)
)

pygame.display.set_caption(
    "Emergent Language — V1.4"
)

clock = pygame.time.Clock()

font = pygame.font.Font(None, 28)

world = World()

agent_a = Agent(world.agent_a)
agent_b = Agent(world.agent_b)

protocol = PredefinedProtocol()

episode = 1
step = 0
successes = 0

# A sends one message at the beginning of the episode
message = protocol.encode(world.target_id)

# B decodes it
decoded_target_id = protocol.decode(message)

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

                message = protocol.encode(
                    world.target_id
                )

                decoded_target_id = protocol.decode(
                    message
                )

                step = 0
                episode += 1

    # --------------------------------
    # B INTERPRETS A'S MESSAGE
    # --------------------------------

    destination = world.target_locations[
        decoded_target_id
    ]

    action = choose_direction(
        agent_b.position,
        destination
    )

    agent_b.move(action)

    step += 1

    # --------------------------------
    # CHECK SUCCESS
    # --------------------------------

    target_distance = distance(
        agent_b.position,
        world.target
    )

    success = target_distance < TARGET_RADIUS

    if success:

        successes += 1

        print(
            f"Episode {episode}: "
            f"SUCCESS in {step} steps | "
            f"Target T{world.target_id} | "
            f"Message {message}"
        )

        world.reset()

        agent_a = Agent(world.agent_a)
        agent_b = Agent(world.agent_b)

        message = protocol.encode(
            world.target_id
        )

        decoded_target_id = protocol.decode(
            message
        )

        episode += 1
        step = 0

    elif step >= MAX_STEPS:

        print(
            f"Episode {episode}: FAILED"
        )

        world.reset()

        agent_a = Agent(world.agent_a)
        agent_b = Agent(world.agent_b)

        message = protocol.encode(
            world.target_id
        )

        decoded_target_id = protocol.decode(
            message
        )

        episode += 1
        step = 0

    # --------------------------------
    # RENDER
    # --------------------------------

    screen.fill((30, 30, 30))

    # Draw all possible target locations
    for i, target_position in enumerate(
        world.target_locations
    ):

        position = world_to_screen(
            target_position
        )

        pygame.draw.circle(
            screen,
            (100, 100, 100),
            position,
            12,
            2
        )

        label = font.render(
            f"T{i}",
            True,
            (150, 150, 150)
        )

        screen.blit(
            label,
            (position[0] + 15, position[1] - 12)
        )

    # Highlight actual target
    pygame.draw.circle(
        screen,
        (255, 220, 80),
        world_to_screen(world.target),
        10
    )

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

    # --------------------------------
    # HUD
    # --------------------------------

    success_rate = (
        successes / max(episode - 1, 1)
    ) * 100

    hud_lines = [
        f"Episode: {episode}",
        f"Step: {step}/{MAX_STEPS}",
        f"Success: {success_rate:.1f}%",
        f"Target: T{world.target_id}",
        f"A -> B: {message}",
        f"B interprets: T{decoded_target_id}",
    ]

    for i, line in enumerate(hud_lines):

        text = font.render(
            line,
            True,
            (230, 230, 230)
        )

        screen.blit(
            text,
            (20, 20 + i * 28)
        )

    pygame.display.flip()

    clock.tick(20)


pygame.quit()