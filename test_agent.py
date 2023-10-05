from multiprocessing import Pool
import gymnasium as gym
import warnings
from functools import partial
from tqdm import tqdm
from agentdqntarget import AgentDQN, AgentDQNTarget
from itertools import product
import utils
from time import time


def class_name(o):
		if isinstance(o,list):
			return "(" + ", ".join([class_name(x) for x in o]) + ")"
		if not o:
			return "None"
		klass = o.__class__
		return full_name(klass)

def full_name(klass):
	return klass.__qualname__

def test_agent_visual(agent):
	env = gym.make("LunarLander-v2", render_mode="human")

	for i in range(10):
		observation, info = env.reset()
		terminated = False
		truncated = False
		while not (terminated or truncated):
			# agent policy that uses the observation and info
			action = agent.getAction(observation)
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				observation, reward, terminated, truncated, info = env.step(action)
	env.close()


def test_agent_stat(agent, verbose=True):
	total_reward = 0
	total_squared_rewards = 0
	env = gym.make("LunarLander-v2")
	for i in range(1000):
		if verbose and i % 100 == 0:
			print(i)
		observation, info = env.reset()
		terminated = False
		truncated = False
		while not (terminated or truncated):
			# agent policy that uses the observation and info
			action = agent.getAction(observation)
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				observation, reward, terminated, truncated, info = env.step(action)
			total_reward += reward
			total_squared_rewards += reward * reward
	env.close()
	return total_reward/1000 , (total_squared_rewards/1000)**(1/2)


def test_agent(agent, mean=True, visual=True, verbose=True):
	if mean and visual:
		pool = Pool()
		# evaluate "solve1(A)" asynchronously
		visual_waiter = pool.apply_async(test_agent_visual, [agent])
		# evaluate "solve2(B)" asynchronously
		mean_waiter = pool.apply_async(test_agent_stat, [agent, verbose])
		visual_waiter.get()
		mean = mean_waiter.get()
		return mean
	elif mean:
		return test_agent_stat(agent, verbose)
	elif visual:
		test_agent_visual(agent)
	else:
		raise ("You should not set both mean and visual to False")

def test_agent_stored_in_state_dict(agent,state_dict, mean=True, visual = True):
	agent.QNN.load_state_dict(state_dict)
	return (state_dict, test_agent(agent, mean=mean, visual=visual, verbose=False))

def compare_agents_stored_in_state_dict_memory(agent, state_memory, mean=True, visual=False):
	agent.eps = 0
	pool = Pool()
	f = partial(test_agent_stored_in_state_dict,agent, mean= mean, visual= visual)
	return list(tqdm(pool.imap_unordered(f, state_memory), total = len(state_memory)))


def floating_average(n):
	return lambda list : [ sum(list[max(0,i-n):i+1])/ min(i+1,n) for i in range(len(list))]

def test_one_learning_rate_and_n_ep(Agent, arguments):
	learning_rate, n_ep = arguments
	eps_decay = 10**(-8/3/learning_rate)
	start = time()
	env = gym.make("LunarLander-v2")
	agent = Agent(
		env, 
		base_lr= learning_rate,
		repr = f"{learning_rate}, {n_ep}"
	)
	
	rewards= agent.runFullLearning(eps_decay= eps_decay,n_ep=n_ep, verbose = False)
	utils.plot_sumrwd_mean_perepi(rewards,floating_average(50)(rewards), file_name=f"pdf/test1/plot_{learning_rate}_{n_ep}")
	learning_time = time() - start
	agent.eps = 0
	mean, sd = test_agent_stat(agent, verbose= False)
	print(
		"learning rate :",learning_rate,
		"\nn_ep :", n_ep,
		"\nlearning time :", learning_time,
		"\nmax recompense en cours d'apprentissage :", max(rewards),
		"\nmoyenne agent final :", mean,
		"\necart-type agent final :", sd,
	)
	return learning_rate, n_ep, mean, sd, learning_time

def test_learning_rate_and_n_ep(use_target, possible_learning, possible_number_of_episodes):
	Agent = AgentDQNTarget if use_target else AgentDQN
	grid = list(product(possible_learning, possible_number_of_episodes))
	loop_core = partial(test_one_learning_rate_and_n_ep, Agent)
	pool = Pool(4)
	return [list(x) for x in tqdm(pool.imap_unordered(loop_core, grid), total = len(grid))]


		

def test_agent_on_target_optim_planner(folder, x):
	Agent, optim, planner_base_lr = x
	planner, base_lr = planner_base_lr
	start = time()
	env = gym.make("LunarLander-v2")
	agent = Agent(
		env,
		planner= planner,
		optim= optim, 
		base_lr = base_lr
	)
	agent.repr = f"{class_name(agent.scheduler)}, {full_name(Agent)}, {full_name(optim)}"
	rewards = agent.runFullLearning(verbose = False)
	utils.plot_sumrwd_mean_perepi(rewards,floating_average(50)(rewards), file_name=f"{folder}/plot_{full_name(Agent)}_{full_name(optim)}_{class_name(agent.scheduler)}")
	learning_time = time() - start
	agent.eps = 0
	mean, sd = test_agent_stat(agent, verbose= False)
	print(
		"learning rate planner:",class_name(agent.scheduler),
		"\nAgent :", full_name(Agent),
		"\noptim :", full_name(optim),
		"\nlearning time :", learning_time,
		"\nmoyenne agent final :", mean,
		"\necart-type agent final :", sd,
	)

	return class_name(agent.scheduler), full_name(Agent), full_name(optim), mean, sd, learning_time

def test_target_optim_planner(folder, possible_Agent, possible_optim, possible_planner):
	grid = product(possible_Agent, possible_optim, possible_planner)
	pool = Pool(5)
	return [list(x) for x in tqdm(pool.imap_unordered(partial(test_agent_on_target_optim_planner, folder), grid), total = len(possible_Agent)*len(possible_optim)*len(possible_planner))]