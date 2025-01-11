import flappy_bird_gymnasium
import gymnasium

env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=True)

obs, _ = env.reset()
print(obs.shape)
print(obs)
while True:
    # Next action:
    # (feed the observation to your agent here)
    action = env.action_space.sample()
    # print(action)

    # Processing:
    obs, reward, terminated, _, info = env.step(action)
    # print(obs)
    frame = env.render()
    print(info)

    # Checking if the player is still alive
    if terminated:
        break

env.close()