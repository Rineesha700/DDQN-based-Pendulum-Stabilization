import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Simple MLP Q-network.

    Matches how dqn_train.py (and demo.py) use it:

        q_net = QNetwork(state_dim, action_dim)
        q_net.apply(init_weights)       # Xavier init on all nn.Linear layers
        q_values = q_net(state_tensor)  # state_tensor: (batch, state_dim)
                                         # returns:      (batch, action_dim)

    state_dim = 6   (dqn_train.py's normalized 6D state)
    action_dim = len(ACTIONS)  (15, from utils.py)
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)
