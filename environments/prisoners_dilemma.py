import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv
import functools


class PrisonersDilemmaEnv(ParallelEnv):
    
    metadata = {"render_modes": ["human"], "name": "prisoners_dilemma_v1"}
    
    def __init__(self, max_rounds=100, observation_window=5):
        super().__init__()
        
        self.max_rounds = max_rounds
        self.observation_window = observation_window
        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = self.possible_agents[:]
        
        # Action space: 0=Cooperate, 1=Defect
        self._action_spaces = {agent: spaces.Discrete(2) for agent in self.agents}
        
        # Observation space: flattened history + round number
        obs_dim = observation_window * 2 + 1
        self._observation_spaces = {
            agent: spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32) 
            for agent in self.agents
        }
        
        # Payoff matrix
        self.payoff_matrix = {
            (0, 0): 3,  # Both cooperate
            (0, 1): 0,  # I cooperate, opponent defects
            (1, 0): 5,  # I defect, opponent cooperates
            (1, 1): 1,  # Both defect
        }
        
        self.reset()
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.agents = self.possible_agents[:]
        self.round_number = 0
        self.action_histories = {agent: [] for agent in self.agents}
        self.cumulative_rewards = {agent: 0 for agent in self.agents}
        
        observations = self._get_observations()
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
    
    def step(self, actions):
        # Store actions
        for agent in self.agents:
            self.action_histories[agent].append(actions[agent])
        
        # Calculate rewards
        rewards = self._calculate_rewards(actions)
        
        # Update cumulative rewards
        for agent in self.agents:
            self.cumulative_rewards[agent] += rewards[agent]
        
        self.round_number += 1
        
        # Check termination
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: self.round_number >= self.max_rounds for agent in self.agents}
        
        if any(truncations.values()):
            self.agents = []
        
        observations = self._get_observations()
        infos = {agent: {'cumulative_reward': self.cumulative_rewards[agent]} for agent in self.agents}
        
        return observations, rewards, terminations, truncations, infos
    
    def _calculate_rewards(self, actions):
        action_0 = actions["agent_0"]
        action_1 = actions["agent_1"]
        
        reward_0 = self.payoff_matrix[(action_0, action_1)]
        reward_1 = self.payoff_matrix[(action_1, action_0)]
        
        return {"agent_0": float(reward_0), "agent_1": float(reward_1)}
    
    def _get_observations(self):
        observations = {}
        
        for agent in self.agents:
            opponent = "agent_1" if agent == "agent_0" else "agent_0"
            
            # Get recent action histories
            own_history = self.action_histories[agent][-self.observation_window:]
            opponent_history = self.action_histories[opponent][-self.observation_window:]
            
            # # Pad with zeros if not enough history
            # while len(own_history) < self.observation_window:
            #     own_history = [0] + own_history
            # while len(opponent_history) < self.observation_window:
            #     opponent_history = [0] + opponent_history
            
            # Create observation vector
            obs = np.array(
                own_history + opponent_history + [self.round_number / self.max_rounds],
                dtype=np.float32
            )
            
            observations[agent] = obs
        
        return observations
    
    def render(self, mode="human"):
        if self.round_number > 0 and self.action_histories["agent_0"]:
            last_idx = len(self.action_histories["agent_0"]) - 1
            action_0 = "C" if self.action_histories["agent_0"][last_idx] == 0 else "D"
            action_1 = "C" if self.action_histories["agent_1"][last_idx] == 0 else "D"
            print(f"Round {self.round_number}: Agent0={action_0}, Agent1={action_1}")
    
    def close(self):
        pass