import random
from collections import deque

import numpy as np


class ReplayBuffer:
    """Fixed-size FIFO experience replay buffer.

    Matches how dqn_train.py uses it:

        buffer = ReplayBuffer()
        buffer.push(state, action_idx, reward, next_state, done)

        (
            states, actions, rewards, next_states, dones
        ) = buffer.sample(BATCH)

        len(buffer)  # checked against MIN_BUFFER

    `state` / `next_state` are expected to already be the normalized 6D
    vectors dqn_train.py works with (post normalize_state()).
    """

    def __init__(self, capacity=100_000, seed=None):

        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

        if seed is not None:
            random.seed(seed)

    def push(self, state, action, reward, next_state, done):

        self.buffer.append((
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size):

        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)
