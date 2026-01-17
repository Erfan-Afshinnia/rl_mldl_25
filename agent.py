import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)


        """
            Critic network
        """
        # TASK 3: critic network for actor-critic algorithm
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic_value = torch.nn.Linear(self.hidden, 1)


        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)


        """
            Critic
        """
        # TASK 3: forward in the critic network
        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        state_value = self.fc3_critic_value(x_critic).squeeze(-1)  # shape: [batch] or []

        return normal_dist, state_value


class Agent(object):
    def __init__(self, policy, device='cpu', baseline_value=None, algo='reinforce'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        self.gamma = 0.99
        self.baseline_value = baseline_value  # used for REINFORCE with baseline
        self.algorithm = algo                 # 'reinforce' or 'actor_critic'

        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []


    def update_policy(self):
        if len(self.action_log_probs) == 0:
            return  # nothing to update

        # Convert stored lists to tensors on the training device
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        # Clear buffers for the next episode
        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []

        if self.algorithm == 'reinforce':
            #
            # TASK 2:
            #   - compute discounted returns
            #   - compute policy gradient loss function given actions and returns
            #   - compute gradients and step the optimizer
            #

            # 1. Discounted returns (Monte Carlo)
            returns = discount_rewards(rewards, self.gamma)  # shape: [T]

            # 2. Apply optional constant baseline
            if self.baseline_value is not None:
                advantages = returns - self.baseline_value
            else:
                advantages = returns

            # 3. Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # 4. Policy gradient loss (REINFORCE)
            policy_loss = -(action_log_probs * advantages).sum()

            # 5. Backpropagation and optimizer step
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

            return policy_loss.item()

        elif self.algorithm == 'actor_critic':
            #
            # TASK 3:
            #   - compute bootstrapped discounted return estimates
            #   - compute advantage terms
            #   - compute actor loss and critic loss
            #   - compute gradients and step the optimizer
            #

            # 1. Get value estimates for states and next_states
            _, values = self.policy(states)          # shape: [T]
            _, next_values = self.policy(next_states)

            # 2. TD(0) targets: r + gamma * V(s') * (1 - done)
            targets = rewards + self.gamma * (1.0 - done) * next_values.detach()

            # 3. Advantages = TD error
            advantages = targets - values

            # Optional: normalize advantages for stability
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # 4. Actor loss (use advantages as weights, detach to stop critic gradients)
            actor_loss = -(action_log_probs * advantages.detach()).sum()

            # 5. Critic loss: MSE between V(s) and targets
            critic_loss = F.mse_loss(values, targets.detach())

            # 6. Total loss (weighting critic loss by 0.5 is common but somewhat arbitrary)
            loss = actor_loss + 0.5 * critic_loss

            # 7. Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return loss.item()
    


    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist, state_value = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob


    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)

