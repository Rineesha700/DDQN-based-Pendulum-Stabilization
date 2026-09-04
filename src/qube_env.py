from quanser.hardware import HIL
import numpy as np
import time


# =========================
# SETTLE PARAMETERS (used by reset(hanging=True))
# =========================
# How long to wait for the pendulum to stop swinging before giving up
# and returning whatever state it's in (safety: never blocks forever).
SETTLE_TIMEOUT = 5.0

# pend_dot (rad/s) below this counts as "settled" for hanging reset.
SETTLE_VEL_THRESH = 0.3

# How often to poll the encoders while waiting to settle.
SETTLE_POLL_INTERVAL = 0.05


# =========================
# helper
# =========================
def wrap_to_pi(theta):

    return (theta + np.pi) % (2 * np.pi) - np.pi



class QubeEnv:

    def __init__(self):

        self.card = HIL(
            "qube_servo3_usb",
            "0"
        )


        # =========================
        # channels
        # =========================

        self.analog_out = np.array(
            [0],
            dtype=np.uint32
        )

        self.encoder_in = np.array(
            [0, 1],
            dtype=np.uint32
        )

        self.other_in = np.array(
            [14000, 14001],
            dtype=np.uint32
        )


        # =========================
        # buffers
        # =========================

        self.analog_buffer = np.zeros(1)

        self.enc_buffer = np.zeros(
            2,
            dtype=np.int32
        )

        self.other_buffer = np.zeros(2)



        # =========================
        # hardware task
        # =========================

        self.task = self.card.task_create_reader(

            1000,

            self.analog_out,
            1,

            self.encoder_in,
            2,

            np.array(
                [0,1,2],
                dtype=np.uint32
            ),
            3,

            self.other_in,
            2
        )


        self.card.task_start(
            self.task,
            0,
            500,
            100000
        )


        self.enable()
        self.balance_counter = 0



    # =========================
    # SAFE ENABLE
    # =========================

    def enable(self):


        # motor voltage zero

        self.card.write_analog(

            np.array(
                [0],
                dtype=np.uint32
            ),

            1,

            np.array(
                [0.0],
                dtype=np.float64
            )
        )


        # enable amplifier

        self.card.write_digital(

            np.array(
                [0],
                dtype=np.uint32
            ),

            1,

            np.array(
                [1],
                dtype=np.int8
            )
        )



    # =========================
    # RESET
    # =========================

    def reset(self, hanging=False):


        # stop motor

        self.card.write_analog(

            np.array(
                [0],
                dtype=np.uint32
            ),

            1,

            np.array(
                [0.0],
                dtype=np.float64
            )
        )


        if hanging:

            # ---------------------------------------------------
            # Wait for the pendulum to physically settle to a
            # low-velocity hanging state before this counts as
            # "reset". Used by evaluate() in dqn_train.py so eval
            # rollouts start from comparable conditions to each
            # other, instead of wherever the arm happened to stop
            # when reset() was called. Motor stays at 0V the whole
            # time (safe - same as the non-hanging path above).
            # ---------------------------------------------------

            settle_deadline = time.time() + SETTLE_TIMEOUT

            while time.time() < settle_deadline:

                self.card.task_read(

                    self.task,

                    1,

                    self.analog_buffer,

                    self.enc_buffer,

                    np.zeros(
                        3,
                        dtype=np.int8
                    ),

                    self.other_buffer
                )

                settled_state = self.get_state()

                pend_dot = settled_state[3]

                if abs(pend_dot) < SETTLE_VEL_THRESH:

                    break

                time.sleep(SETTLE_POLL_INTERVAL)

        else:

            # read current state

            self.card.task_read(

                self.task,

                1,

                self.analog_buffer,

                self.enc_buffer,

                np.zeros(
                    3,
                    dtype=np.int8
                ),

                self.other_buffer
            )

        self.balance_counter = 0
        return self.get_state()



    # =========================
    # STEP
    # =========================

    def step(self, action):


        # =================================================
        # ACTION HANDLING
        # =================================================
        #
        # ACTIONS already contains voltages:
        #
        # [-3.5 ... +3.5]
        #
        # Do NOT multiply again.
        #

        MAX_VOLT = 3.5


        action = np.clip(

            action,

            -MAX_VOLT,

            MAX_VOLT
        )



        # apply voltage

        self.card.write_analog(

            np.array(
                [0],
                dtype=np.uint32
            ),

            1,

            np.array(
                [action],
                dtype=np.float64
            )
        )



        # read hardware

        self.card.task_read(

            self.task,

            1,

            self.analog_buffer,

            self.enc_buffer,

            np.zeros(
                3,
                dtype=np.int8
            ),

            self.other_buffer
        )



        state = self.get_state()



        motor = state[0]

        pend = state[1]

        motor_dot = state[2]

        pend_dot = state[3]



        # UPRIGHT ERROR
        upright_error = wrap_to_pi(
            pend - np.pi
        )


        # REWARD FUNCTION
        # =================================================
        reward = 0.0

        # Swing-up reward
        reward += (20.0 *np.cos(upright_error))
        # angle penalty
        reward -= (1.0 *upright_error ** 2)
        # Reduce movement
        near_upright_weight = np.exp(-5.0 *upright_error ** 2)

        reward -= (0.03 *pend_dot ** 2)
        reward -= (0.9 *near_upright_weight *pend_dot ** 2)
        reward -= (0.05 *motor_dot ** 2)
        reward -= (0.02 *motor ** 2)

        # control penalty
        reward -= (0.003 *action ** 2)

        stability_bonus = (
        80.0 *
        np.exp(-12.0 * upright_error ** 2) *
        np.exp(-0.3 * pend_dot ** 2) *
        np.exp(-0.1 * motor_dot ** 2)

    )

        reward += stability_bonus


        if (abs(upright_error) < 0.10 and abs(pend_dot) < 0.5 and abs(motor_dot) < 0.5):

            self.balance_counter = min(self.balance_counter + 1,100)
            reward += 40 + 0.2 * self.balance_counter

        else:
            self.balance_counter = max(0,self.balance_counter - 1)

      
        # Soft arm limit penalty
    
        # Start penalizing when the arm approaches the mechanical limit
        if abs(motor) > 2.0:
            reward -= 10 * (abs(motor) - 2.0)

        # Extra penalty very close to the end stop
        #if abs(motor) > 2.3:
            #reward -= 30

        

        # =================================================
        # TERMINATION
        # =================================================

        done = (

            abs(motor) > 2.5

            or

            abs(pend_dot) > 35

        )



        return state, reward, done
    
    # =========================
    # STATE (6D)
    # =========================

    def get_state(self):


        # QUBE Servo 3 encoder resolution
        RAD_PER_COUNT = (

            2 * np.pi

        ) / 2048



        # encoder positions

        motor = (

            self.enc_buffer[0]

            *

            RAD_PER_COUNT

        )


        pend = (

            self.enc_buffer[1]

            *

            RAD_PER_COUNT

        )



        # wrap angles

        motor = wrap_to_pi(
            motor
        )

        pend = wrap_to_pi(
            pend
        )



        # velocity channels

        motor_dot = (

            self.other_buffer[0]

            *

            RAD_PER_COUNT

        )


        pend_dot = (

            self.other_buffer[1]

            *

            RAD_PER_COUNT

        )



        return np.array(

            [

                motor,

                pend,

                motor_dot,

                pend_dot,

                np.sin(pend),

                np.cos(pend)

            ],

            dtype=np.float32

        )



    # =========================
    # CLOSE SAFE
    # =========================

    def close(self):


        # stop motor voltage

        self.card.write_analog(

            np.array(
                [0],
                dtype=np.uint32
            ),

            1,

            np.array(
                [0.0],
                dtype=np.float64
            )
        )



        # stop hardware task

        self.card.task_stop(
            self.task
        )


        self.card.task_delete(
            self.task
        )


        self.card.close()