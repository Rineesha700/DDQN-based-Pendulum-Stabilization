import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random
import os


from dqn_model import QNetwork
from replay_buffer import ReplayBuffer
from qube_env import QubeEnv
from utils import ACTIONS, wrap_to_pi

# RANDOM SEED


SEED = 42


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# Hyperparameters

GAMMA = 0.99
LR = 1e-4
BATCH = 128


# Exploration
EPS = 0.05
EPS_DECAY = 0.99995
EPS_MIN = 0.05

# Training
TARGET_TAU = 0.005
TOTAL_STEPS = 70000

# Replay
MIN_BUFFER = 5000

# Checkpoint
SAVE_INTERVAL = 5000
MODEL_NAME = "dqn_qube.pth"
# MODEL_NAME = "best_dqn_qube.pth"


# ==========================================================
# Evaluation (logic ported from train.py)
# ==========================================================
# train.py does not save "best" based on a single noisy training
# episode's raw reward - DQN training is not monotonic, so a
# training episode can peak on luck (exploration noise, a good
# random seed of initial conditions) and then get worse. Instead
# it periodically freezes exploration (epsilon=0), runs several
# greedy rollouts, and scores the checkpoint on how much of the
# time it actually held upright. That's what EVAL_EVERY /
# EVAL_EPISODES / EVAL_MAX_STEPS / UPRIGHT_THRESH drive below.

EVAL_EVERY = 5000
EVAL_EPISODES = 10


EVAL_MAX_STEPS = 2000

# Same "how close to vertical counts as upright" definition used for
# the reward shaping in qube_env.step(), used here only to score
# evaluation rollouts.
UPRIGHT_THRESH = 0.5

# ==========================================================
# State normalization
# ==========================================================

STATE_SCALE = np.array([np.pi,np.pi,20.0,20.0,1.0,1.0], dtype=np.float32)   #[motor angle, pend angle, motor velocity, pend velocity, sin(pend angle), cos(pend angle)]
def normalize_state(state):
    return (state / STATE_SCALE).astype(np.float32)

# Environment
env = QubeEnv()
state_dim = 6
action_dim = len(ACTIONS)

# Networks
q_net = QNetwork(state_dim,action_dim)
target_net = QNetwork(state_dim,action_dim)

# ==========================================================
# Resume from previous best checkpoint (warm start), or init fresh
# ==========================================================
RESUME_FROM =  "best_dqn_qube.pth"   # set to None to force training from scratch

# Xavier Initialization
def init_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)

if RESUME_FROM is not None and os.path.exists(RESUME_FROM):
    print(f"Resuming from checkpoint: {RESUME_FROM}")
    q_net.load_state_dict(torch.load(RESUME_FROM, map_location="cpu"))
    resumed = True
else:
    print("No checkpoint found - training q_net from scratch (Xavier init)")
    q_net.apply(init_weights)
    resumed = False

target_net.load_state_dict(q_net.state_dict())
target_net.eval()

# Exploration schedule: resumed models already know how to act, so don't
# blow that away with fully-random actions the way a from-scratch run needs.
EPS = 0.08 if resumed else EPS
EPS_DECAY = 0.9997 if resumed else EPS_DECAY

# Optimizer
opt = optim.Adam(q_net.parameters(),lr=LR)
buffer = ReplayBuffer()

# ACTION SELECTION
def select_action(state):
    global EPS
    if np.random.random() < EPS:
        return np.random.randint(action_dim)

    state_t = torch.FloatTensor(state ).unsqueeze(0)

    with torch.no_grad():

        q_values = q_net(state_t)


    return torch.argmax(q_values).item()


# SOFT TARGET UPDATE
def soft_update():

    for target_param, param in zip( target_net.parameters(), q_net.parameters()):

        target_param.data.copy_(TARGET_TAU * param.data+(1.0 - TARGET_TAU)*target_param.data)


# EVALUATION (greedy rollouts, ported from train.py's evaluate())
def evaluate():
   
    returns = []
    upright_fracs = []
    swung_up_flags = []

    for _ in range(EVAL_EPISODES):

        eval_state = env.reset()
        eval_state = normalize_state(eval_state)

        total = 0.0
        up_steps = 0
        got_up = 0

        for _ in range(EVAL_MAX_STEPS):

            eval_state = np.clip(eval_state, -5, 5)
            state_t = torch.FloatTensor(eval_state).unsqueeze(0)

            with torch.no_grad():
                q_values = q_net(state_t)

            action_idx = torch.argmax(q_values).item()
            action = np.array([ACTIONS[action_idx]], dtype=np.float64)

            next_state, reward, done = env.step(action[0])

            upright_error = abs(wrap_to_pi(next_state[1] - np.pi))
            is_upright = int(upright_error < UPRIGHT_THRESH)
            up_steps += is_upright
            got_up = max(got_up, is_upright)

            total += reward
            eval_state = normalize_state(next_state)

            if done:
                break

        returns.append(total)
        upright_fracs.append(up_steps / EVAL_MAX_STEPS)
        swung_up_flags.append(got_up)

    return float(np.mean(returns)), float(np.mean(upright_fracs)), float(np.mean(swung_up_flags))

print("Starting Double DQN training on QUBE...")


# RESET

state = env.reset()
state = normalize_state(state)

episode_reward = 0

episode = 0
best_reward = -float("inf")

# History + best-by-evaluation tracking (ported from train.py)
history = {
    "episode_rewards": [],
    "losses": [],
    "eval_steps": [],
    "eval_rewards": [],
    "eval_upright_frac": [],
}
best_score = -1e9


if resumed:
    print("Evaluating resumed checkpoint before continuing training...")
    base_r, base_upright, base_swung = evaluate()
    best_score = base_upright + 0.5 * base_swung
    print(f"Baseline eval score = {best_score:.3f} "
          f"(upright={100*base_upright:.1f}%, swung_up={100*base_swung:.0f}%)")
    state = env.reset()
    state = normalize_state(state)
    episode_reward = 0



# TRAINING LOOP

for step in range(TOTAL_STEPS):

    # Select action
    state = np.clip(state, -5, 5)
    action_idx = select_action( state)
    action = np.array([ACTIONS[action_idx]],dtype=np.float64)

    # Environment step
    next_state, reward, done = env.step(action[0])

    # Normalize next state
    next_state = normalize_state(next_state)

    # Reward clipping
    reward = np.clip(reward,-100,100)

    # Store transition
    buffer.push(state,action_idx,reward,next_state,done)
    episode_reward += reward

    # Episode reset
    if done:

        print(f"Episode {episode} reward: {episode_reward:.2f}")
        history["episode_rewards"].append(episode_reward)

       
        if episode_reward > best_reward:
            best_reward = episode_reward
            print(f"New best training-episode reward = {best_reward:.2f} (not saved - see eval-based checkpointing)")

        episode += 1
        episode_reward = 0
        state = env.reset()
        state = normalize_state(state)

    else:
        state = next_state
    
    # LEARNING
    if len(buffer) >= MIN_BUFFER:


        (states,actions,rewards,next_states,dones) = buffer.sample(BATCH)
        states = torch.FloatTensor(states)

        next_states = torch.FloatTensor(next_states)

        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)

        dones = torch.FloatTensor(dones)
      
        # Current Q values
        current_q = q_net(states).gather(1,actions.unsqueeze(1)).squeeze(1)

        # Double DQN target
        #
        # Action selection:
        # q_net
        #
        # Action evaluation:
        # target_net
        with torch.no_grad():


            next_actions = torch.argmax(

                q_net(next_states),

                dim=1

            )


            next_q = target_net(

                next_states

            ).gather(

                1,

                next_actions.unsqueeze(1)

            ).squeeze(1)



            target_q = (

                rewards

                +

                GAMMA

                *

                next_q

                *

                (1 - dones)

            )



        # --------------------------------------------------
        # Huber loss
        # --------------------------------------------------

        loss = nn.SmoothL1Loss()(

            current_q,

            target_q

        )



        # --------------------------------------------------
        # Optimization
        # --------------------------------------------------

        opt.zero_grad()


        loss.backward()



        # Gradient clipping

        torch.nn.utils.clip_grad_norm_(

            q_net.parameters(),

            1.0

        )


        # Single optimizer step

        opt.step()

        history["losses"].append(loss.item())



        # --------------------------------------------------
        # Soft target update
        # --------------------------------------------------

        soft_update()



    # ======================================================
    # EPSILON DECAY
    # ======================================================

    EPS = max(

        EPS_MIN,

        EPS * EPS_DECAY

    )



    # ======================================================
    # Logging
    # ======================================================

    if step % 200 == 0:


        print(

            f"Step: {step} | "

            f"EPS: {EPS:.3f} | "

            f"Reward: {episode_reward:.2f} | "

            f"Buffer: {len(buffer)}"

        )



    # ======================================================
    # Evaluation-based checkpointing (ported from train.py)
    # ======================================================

    if step % EVAL_EVERY == 0 and step > 0:

        mean_r, upright, swung_up = evaluate()

        history["eval_steps"].append(step)
        history["eval_rewards"].append(mean_r)
        history["eval_upright_frac"].append(upright)

        print(
            f"[EVAL] step {step}/{TOTAL_STEPS} | "
            f"return={mean_r:.2f} | "
            f"upright={100*upright:.1f}% | "
            f"swung_up={100*swung_up:.0f}%"
        )

        # Reward both swinging up and holding, same as train.py
        score = upright + 0.5 * swung_up

        if score > best_score:
            best_score = score
            torch.save(q_net.state_dict(), "best_dqn_qube.pth")
            print(f"New best model saved! Eval score = {best_score:.3f}")

        # evaluate() ran its own rollouts on the live env, so the
        # in-progress training episode's trajectory is no longer valid -
        # resume training from a fresh reset.
        state = env.reset()
        state = normalize_state(state)
        episode_reward = 0

    # ======================================================
    # Checkpoint
    # ======================================================

    if step % SAVE_INTERVAL == 0 and step > 0:


        torch.save(

            q_net.state_dict(),

            f"dqn_qube_{step}.pth"

        )


        print(

            f"Checkpoint saved at step {step}"

        )



# ==========================================================
# FINISH
# ==========================================================

env.close()



torch.save(

    q_net.state_dict(),

    MODEL_NAME

)


np.savez(
    "history_dqn_qube.npz",
    **{k: np.array(v, dtype=object) for k, v in history.items()}
)


print(

    "Training finished"

)

print(f"\nDone. best eval score = {best_score:.3f}")
print("Best model    -> best_dqn_qube.pth")
print("Final model   -> " + MODEL_NAME)
print("History       -> history_dqn_qube.npz")
