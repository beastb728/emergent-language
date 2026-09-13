import pygame

from environment.world import World


WIDTH = 800
HEIGHT = 600

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Emergent Language")

clock = pygame.time.Clock()

world = World(WIDTH, HEIGHT)

running = True

while running:

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                world.reset()

    screen.fill((30, 30, 30))

    # Agent A
    pygame.draw.circle(
        screen,
        (80, 160, 255),
        world.agent_a.astype(int),
        15
    )

    # Agent B
    pygame.draw.circle(
        screen,
        (255, 100, 100),
        world.agent_b.astype(int),
        15
    )

    # Target
    pygame.draw.circle(
        screen,
        (255, 220, 80),
        world.target.astype(int),
        10
    )

    pygame.display.flip()

    clock.tick(60)

pygame.quit()