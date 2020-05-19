import gym
env = gym.make('SpaceInvaders-v0')
print(env.action_space)
print(env.observation_space)
for i_episode in range(2):
    observation = env.reset()
    for t in range(100):
        # if t % 100 == 0:
        #     print(observation)

        env.render()
        action = env.action_space.sample()
        # print(action)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()