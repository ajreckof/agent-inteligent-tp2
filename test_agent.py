from multiprocessing import Pool
import gymnasium as gym


def test_agent_visual(agent):
    env = gym.make("LunarLander-v2", render_mode="human")

    for i in range(10):
        observation, info = env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            # agent policy that uses the observation and info
            action = agent.getAction(observation)
            observation, reward, terminated, truncated, info = env.step(action)
    env.close()


def test_agent_mean(agent):
    total_reward = 0
    env = gym.make("LunarLander-v2")
    for i in range(1000):
        if i % 100 == 0:
            print(i)
        observation, info = env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            # agent policy that uses the observation and info
            action = agent.getAction(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
    env.close()
    return total_reward/1000


def test_agent(agent, mean=True, visual=True):
    if mean and visual:
        pool = Pool()
        # evaluate "solve1(A)" asynchronously
        visual_waiter = pool.apply_async(test_agent_visual, [agent])
        # evaluate "solve2(B)" asynchronously
        mean_waiter = pool.apply_async(test_agent_mean, [agent])
        visual_waiter.get()
        mean = mean_waiter.get()
        return mean
    elif mean:
        return test_agent_mean()
    elif visual:
        test_agent_visual()
    else:
        raise ("You should not set both mean and visual to False")
