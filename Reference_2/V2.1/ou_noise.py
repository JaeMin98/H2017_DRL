import numpy as np
import random
import copy
import config

class OUNoise:
    """
    Ornstein-Uhlenbeck process.
    Used to add time-correlated noise to the actions for exploration.
    """

    def __init__(self, size, seed, mu=config.MU, theta=config.THETA, sigma=config.SIGMA):
        """
        Initialize parameters and noise process.
        
        Args:
        size (int): The dimension of the noise
        seed (int): Random seed for reproducibility
        mu (float): The mean of the noise (default from config)
        theta (float): The speed of mean reversion (default from config)
        sigma (float): The volatility or size of the noise (default from config)
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """
        Update internal state and return it as a noise sample.
        
        Returns:
        np.array: The current state of the noise process
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state