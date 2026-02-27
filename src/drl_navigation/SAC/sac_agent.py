"""
Soft Actor-Critic algorithm implementation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleQCritic(nn.Module):
    """Twin Q-network for SAC"""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=256, hidden_depth=2):
        super().__init__()
        
        # Q1 network
        self.q1_layers = nn.ModuleList()
        self.q1_layers.append(nn.Linear(obs_dim + action_dim, hidden_dim))
        for _ in range(hidden_depth - 1):
            self.q1_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.q1_out = nn.Linear(hidden_dim, 1)
        
        # Q2 network  
        self.q2_layers = nn.ModuleList()
        self.q2_layers.append(nn.Linear(obs_dim + action_dim, hidden_dim))
        for _ in range(hidden_depth - 1):
            self.q2_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.q2_out = nn.Linear(hidden_dim, 1)
        
        self.activation = nn.ReLU()
        
    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        
        # Q1 forward
        q1 = x
        for layer in self.q1_layers:
            q1 = self.activation(layer(q1))
        q1 = self.q1_out(q1)
        
        # Q2 forward
        q2 = x
        for layer in self.q2_layers:
            q2 = self.activation(layer(q2))
        q2 = self.q2_out(q2)
        
        return q1, q2

class DiagGaussianActor(nn.Module):
    """Gaussian policy network for SAC"""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=256, hidden_depth=2, log_std_bounds=[-5, 2]):
        super().__init__()
        self.action_dim = action_dim
        self.log_std_bounds = log_std_bounds
        
        # Shared trunk
        self.trunk_layers = nn.ModuleList()
        self.trunk_layers.append(nn.Linear(obs_dim, hidden_dim))
        for _ in range(hidden_depth - 1):
            self.trunk_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Output heads
        self.mu_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
        self.activation = nn.ReLU()
        
    def forward(self, obs):
        x = obs
        for layer in self.trunk_layers:
            x = self.activation(layer(x))
            
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.tanh(log_std)
        
        # Apply log std bounds
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        
        std = torch.exp(log_std)
        
        return torch.distributions.Normal(mu, std)

class SAC:
    """Soft Actor-Critic implementation"""
    
    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        max_action=1.0,
        discount=0.99,
        tau=0.005,
        alpha_lr=3e-4,
        actor_lr=3e-4,
        critic_lr=3e-4,
        learnable_temperature=True,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.discount = discount
        self.tau = tau
        self.max_action = max_action
        self.learnable_temperature = learnable_temperature
        
        # Networks
        self.critic = DoubleQCritic(state_dim, action_dim).to(device)
        self.critic_target = DoubleQCritic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor = DiagGaussianActor(state_dim, action_dim).to(device)
        
        # Temperature (alpha)
        self.log_alpha = torch.tensor(np.log(0.1)).to(device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -action_dim
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        
        self.train()
        
    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def act(self, obs, sample=True):
        """Get action from policy"""
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
            dist = self.actor(obs)
            action = dist.sample() if sample else dist.mean
            action = action.clamp(-self.max_action, self.max_action)
            return action.cpu().numpy()[0]
    
    def update(self, replay_buffer, batch_size=256):
        """Single SAC update step"""
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        
        # Update critic
        with torch.no_grad():
            next_dist = self.actor(next_states)
            next_actions = next_dist.rsample()
            next_log_probs = next_dist.log_prob(next_actions).sum(-1, keepdim=True)
            
            target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * next_log_probs
            target_Q = rewards + (1 - dones) * self.discount * target_V
        
        current_Q1, current_Q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        dist = self.actor(states)
        new_actions = dist.rsample()
        log_probs = dist.log_prob(new_actions).sum(-1, keepdim=True)
        
        actor_Q1, actor_Q2 = self.critic(states, new_actions)
        actor_Q = torch.min(actor_Q1, actor_Q2)
        
        actor_loss = (self.alpha.detach() * log_probs - actor_Q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha
        if self.learnable_temperature:
            alpha_loss = (self.alpha * (-log_probs - self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        # Update target critic
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha.item(),
            'average_reward': rewards.mean().item()
        }
    
    def save(self, path):
        """Save model weights"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha,
        }, path)
    
    def load(self, path):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.log_alpha = checkpoint['log_alpha']