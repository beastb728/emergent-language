from environment.compositional_world_v2 import CompositionalWorldV2


world = CompositionalWorldV2()

for i in range(10):
    world.reset()

    state = world.sender_state()

    receiver_state = world.receiver_state(
        symbol_x=1,
        symbol_y=2
    )

    print(
        f"Episode {i}: "
        f"X={world.x_region}, "
        f"Y={world.y_region}, "
        f"target={world.target}, "
        f"sender_state={len(state)}, "
        f"receiver_state={len(receiver_state)}, "
        f"initially_reached={world.in_target_region()}"
    )