## Question: How do you code a Reinforcement Learning Network in Pytorch?


# 🕹️ Reinforcement Learning with PyTorch

This example shows a simple Q-Learning-based DQN (Deep Q Network).

## 🧪 DQN Agent for OpenAI Gym

```python
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import random
import numpy as np
from collections import deque

env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.model(x)

q_net = DQN()
optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
criterion = nn.MSELoss()
replay_buffer = deque(maxlen=10000)
gamma = 0.99

for episode in range(10):
    state = env.reset()
    total_reward = 0
    for t in range(200):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = q_net(state_tensor)
        action = q_values.argmax().item() if random.random() > 0.1 else env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        if done:
            break

        if len(replay_buffer) >= 64:
            batch = random.sample(replay_buffer, 64)
            states, actions, rewards, next_states, dones = zip(*batch)
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones).unsqueeze(1)

            q_values = q_net(states).gather(1, actions)
            next_q = q_net(next_states).max(1, keepdim=True)[0].detach()
            target_q = rewards + gamma * next_q * (1 - dones)

            loss = criterion(q_values, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(f"Episode {episode+1}, Total Reward: {total_reward}")
```
