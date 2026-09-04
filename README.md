# Double DQN Control of a Rotary Inverted Pendulum

A reinforcement learning project implementing a **Double Deep Q-Network (Double DQN)** controller for stabilizing a rotary inverted pendulum in the upright position using the **Quanser QUBE-Servo 3**.

## Project Overview

The objective of this project is to develop and deploy a reinforcement learning controller capable of maintaining the pendulum in the **upright (above-vertical) position** through real-time control of the QUBE-Servo 3.

The controller uses **Double DQN** to learn the relationship between the pendulum's state and discrete motor-voltage actions while reducing the Q-value overestimation associated with standard DQN.

## Key Features

* Implemented a **Double DQN** reinforcement learning controller.
* Designed a state representation using pendulum angle, pendulum angular velocity, motor position, and motor angular velocity.
* Used a **discrete voltage action space** for motor control.
* Implemented **experience replay** and a separate target network.
* Used **ε-greedy exploration** during training.
* Deployed the trained policy for **real-time control on the Quanser QUBE-Servo 3**.
* Evaluated stabilization using quantitative control metrics such as angle error, stabilization duration, and successful trials.

## System

**Hardware:** Quanser QUBE-Servo 3
**Controller:** Double Deep Q-Network
**Learning:** Reinforcement Learning
**Framework:** PyTorch
**Programming Language:** Python

## State Representation

The controller observes a four-dimensional state:

```text
[pendulum angle,
 pendulum angular velocity,
 motor position,
 motor angular velocity]
```

These measurements are obtained from the QUBE-Servo 3 encoders and used as the input to the neural network.

## Action Space

The controller uses discrete motor-voltage commands:

```text
Negative voltage → Rotate in one direction
Zero voltage     → No motor command
Positive voltage → Rotate in the opposite direction
```

The discrete action space allows the DQN to select the control action with the highest estimated Q-value.

## Double DQN

The Double DQN architecture consists of:

* **Online Q-network** — selects the best next action.
* **Target Q-network** — estimates the value of the selected action.
* **Replay buffer** — stores previous transitions for randomized training.
* **ε-greedy policy** — balances exploration and exploitation.

Compared with standard DQN, Double DQN separates **action selection** from **action evaluation**, helping reduce Q-value overestimation.

## Reward Function

The reward encourages the pendulum to remain close to the upright position while limiting excessive motion and control effort.

The main components include:

* Upright-angle reward
* Pendulum velocity penalty
* Motor position penalty
* Motor velocity penalty
* Voltage/control-effort penalty

This encourages the learned policy to maintain the pendulum near the upright position with smooth control.

## Training Process

The training loop follows the standard reinforcement learning workflow:

```text
Initialize environment
        ↓
Observe state
        ↓
Select action using ε-greedy policy
        ↓
Apply motor voltage
        ↓
Observe next state and reward
        ↓
Store transition in replay buffer
        ↓
Sample mini-batch
        ↓
Update online Q-network
        ↓
Update target network
        ↓
Decay exploration rate
```



## Project Structure

```text
QUBE-Double-DQN/
│
├── qube_env.py              # Environment and state/reward definitions
├── dqn_model.py              # Double DQN agent and neural network
├── dqn_train.py            # Training loop
├── replay_buffer         # state replay
```








## Technologies

* Python
* PyTorch
* NumPy
* Quanser QUBE-Servo 3
* Deep Reinforcement Learning
* Double DQN
* Experience Replay
* Real-Time Control

```



