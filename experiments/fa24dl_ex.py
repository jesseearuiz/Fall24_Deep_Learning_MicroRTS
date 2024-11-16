import numpy as np
import torch
from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from DeepQ import Agent, DeepQNetwork

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


# Initialize the environment and agent
envs = MicroRTSGridModeVecEnv(
    num_selfplay_envs=2,
    num_bot_envs=3,
    max_steps=1000,
    ai2s=[microrts_ai.coacAI, microrts_ai.passiveAI, microrts_ai.randomAI],
    map_paths=["maps/16x16/basesWorkers16x16.xml"],
)
envs.action_space.seed(0)
envs.reset()
net = DeepQNetwork(0.0001, [np.prod(envs.observation_space.shape)], 256, 256,output_size=envs.height * envs.width * np.sum(envs.action_space_dims))
agents = [Agent(gamma=0.99, epsilon=1.0, lr=0.0001, input_size=[np.prod(envs.observation_space.shape)],
			  batch_size=64, output_size=envs.height * envs.width * np.sum(envs.action_space_dims), max_mem_size=1000000,
			  min_epsilon=0.01, epsilon_decrement=5e-5,network=net) for _ in range(envs.num_envs)]
# Training loop
scores = []
num_games = 1000
action = np.zeros((envs.num_envs, envs.height * envs.width, sum(envs.action_space_dims)))
for i in range(num_games):
	obs = envs.reset()
	obs = torch.tensor(obs, dtype=torch.float32)
	score = 0
	for j in range(envs.max_steps): 
		#get action mask
		mask = envs.get_action_mask()
		mask = mask.reshape(-1, mask.shape[-1])
		mask[mask == 0] = -9e8
		#shape mask and action to be of shape (num_envs, height*width, sum(action_space_dims))
		mask = torch.tensor(mask, dtype=torch.float32).reshape(envs.num_envs, envs.height * envs.width, sum(envs.action_space_dims))
		j = 0
		for agent in agents:
			action[j] = agent.choose_action(obs[j]).reshape(envs.height*envs.width,-1)
			j += 1
		action = action.reshape(envs.num_envs,envs.height * envs.width,sum(envs.action_space_dims))
		masked_action = torch.tensor(action, dtype=torch.float32) + mask
		masked_action = torch.argmax(masked_action, dim=2)
		print(masked_action)
		new_obs, reward, done, info = envs.step(action)
		score += sum(reward)
		j = 0
		for agent in agents:
			agent.store_transition(obs[j].flatten(),action[j].flatten(),reward[j].flatten(),new_obs[j].flatten(),done[j])
			j += 1
		obs = new_obs
		agent.learn()
	
	scores.append(score)

	avg_score = np.mean(scores[-50:])

	print("Episode: ", i, " Score: ", score, " Average Score: ",
			avg_score, " Epsilon: ", agent.epsilon)

envs.close()