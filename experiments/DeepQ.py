import torch as T
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy

class DeepQNetwork(nn.Module):
    def __init__(self,lr,input_size,layer_1_size,layer_2_size,output_size):
        super(DeepQNetwork,self).__init__()
        self.lr = lr
        self.input_layer = nn.Linear(*input_size,layer_1_size,dtype=T.float32)
        self.layer_1 = nn.Linear(layer_1_size,layer_2_size,dtype=T.float32)
        self.layer_2 = nn.Linear(layer_2_size,output_size,dtype=T.float32)
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cpu')
        self.to(self.device)

    def forward(self,state):
        x = F.relu(self.input_layer(state))
        x = F.relu(self.layer_1(x))
        outputs = self.layer_2(x)
        return outputs
       
class Agent(): 
    def __init__(self, gamma, epsilon, lr, input_size, batch_size, output_size,
                  max_mem_size, min_epsilon, epsilon_decrement, network):
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.input_size = input_size
        self.action_space = [i for i in range(output_size)]
        self.batch_size = batch_size
        self.mem_size = max_mem_size
        self.min_epsilon = min_epsilon
        self.epsilon_decrement = epsilon_decrement
        self.memory_ctr = 0
        self.output_dim = output_size

        self.Q_eval = network
        #Initializing memory
        self.state_memory = np.zeros((self.mem_size, *self.input_size),dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *self.input_size),dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size,self.output_dim),dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size,dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size,dtype=np.bool_)
        self.training_network = self.Q_eval

    def store_transition(self, state, action, reward, new_state, done):
        index = self.memory_ctr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.memory_ctr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(np.array([observation])).to(self.Q_eval.device).flatten()
            action = self.Q_eval.forward(state)
        else: 
            action = np.random.random_sample(self.output_dim)
        
        return action
    
    def learn(self):
        if self.memory_ctr < self.batch_size:
            return
        
        if self.memory_ctr % 250 == 0:
            self.training_network == copy.deepcopy(self.Q_eval)
        #zeros out gradients
        self.Q_eval.optimizer.zero_grad()
        #determines how much of the memory can be sampled
        max_mem = min(self.memory_ctr, self.mem_size)
        #samples batch_size number of memories 
        batch = np.random.choice(max_mem, self.batch_size, replace=True)

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        action_batch = T.tensor(self.action_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device) 

        q_eval = self.Q_eval.forward(state_batch)[batch_index]
        q_next = self.training_network.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma*T.max(q_next, dim=1)[0]
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()

        self.Q_eval.optimizer.step()

        if self.epsilon > self.min_epsilon:
            self.epsilon -= self.epsilon_decrement
        else:
            self.epsilon = self.min_epsilon