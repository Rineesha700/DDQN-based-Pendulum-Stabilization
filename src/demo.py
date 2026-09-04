"""Run a trained DQN checkpoint on the QUBE hardware for a live demo.

    python demo.py                          # uses best_dqn_qube.pth
    python demo.py --model dqn_qube.pth      # use a different checkpoint
    python demo.py --episodes 3
    python demo.py --hanging                 # settle to a hang before each run

Ctrl+C safely stops the motor and releases the hardware before exiting.

No exploration here (this is greedy / epsilon=0 inference only) - if you
want to reproduce the training-time evaluation metric instead, use
evaluate() in dqn_train.py.
"""

import argparse
import time

import numpy as np
import torch

from dqn_model import QNetwork
from qube_env import QubeEnv
from utils import ACTIONS, wrap_to_pi

# ==========================================================
# Must match dqn_train.py exactly - this is what the network
# was trained on. Do not change independently of dqn_train.py.
# ==========================================================

STATE_SCALE = np.array([np.pi, np.pi, 20.0, 20.0, 1.0, 1.0], dtype=np.float32)


def normalize_state(state):
    return (state / STATE_SCALE).astype(np.float32)


# Same "how close to vertical counts as upright" threshold used for
# reporting in dqn_train.py's evaluate().
UPRIGHT_THRESH = 0.25


def load_policy(model_path, state_dim, action_dim):

    q_net = QNetwork(state_dim, action_dim)
    q_net.load_state_dict(torch.load(model_path, map_location="cpu"))
    q_net.eval()

    return q_net


def act(q_net, state):

    # Same clipping dqn_train.py applies before every action selection.
    state = np.clip(state, -5, 5)

    state_t = torch.FloatTensor(state).unsqueeze(0)

    with torch.no_grad():
        q_values = q_net(state_t)

    return torch.argmax(q_values).item()


def run_episode(env, q_net, max_steps, hanging):

    state = env.reset(hanging=hanging)
    state = normalize_state(state)

    total_reward = 0.0
    upright_steps = 0

    for t in range(max_steps):

        action_idx = act(q_net, state)
        action = np.array([ACTIONS[action_idx]], dtype=np.float64)

        next_state, reward, done = env.step(action[0])

        upright_error = abs(wrap_to_pi(next_state[1] - np.pi))
        if upright_error < UPRIGHT_THRESH:
            upright_steps += 1

        total_reward += reward
        state = normalize_state(next_state)

        if done:
            print(f"  step {t}: episode ended early (safety limit hit)")
            break

    return total_reward, upright_steps / max_steps


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="best_dqn_qube.pth")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument(
        "--hanging",
        action="store_true",
        help="settle to a hanging start before each run (see qube_env.reset)",
    )
    args = parser.parse_args()

    state_dim = 6
    action_dim = len(ACTIONS)

    print(f"Loading policy from {args.model} ...")
    q_net = load_policy(args.model, state_dim, action_dim)

    env = QubeEnv()

    try:
        for ep in range(args.episodes):

            print(f"\nDemo run {ep + 1}/{args.episodes}")

            total_reward, upright_frac = run_episode(
                env, q_net, args.max_steps, args.hanging
            )

            print(f"  return={total_reward:.2f}  upright={100 * upright_frac:.1f}%")

            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\nInterrupted - stopping motor safely.")

    finally:
        env.close()
        print("Motor stopped, hardware released.")


if __name__ == "__main__":
    main()
