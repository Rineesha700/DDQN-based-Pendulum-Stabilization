import numpy as np

# ENCODER CONVERSION
RAD_PER_COUNT = 2 * np.pi / (512 * 4)

ACTIONS = [-3.5,-3.0,-2.5,-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5]
#ACTIONS = [-2.5,-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,2.5]


def wrap_to_pi(theta):

    return (theta + np.pi) % (2 * np.pi) - np.pi


def state_from_encoder(enc_motor,enc_pend,vel_motor,vel_pend):
    theta_m = wrap_to_pi(enc_motor * RAD_PER_COUNT)
    theta_p = wrap_to_pi(enc_pend * RAD_PER_COUNT)

    return np.array([theta_m,theta_p,vel_motor,vel_pend,np.sin(theta_p),np.cos(theta_p) ], dtype=np.float32)