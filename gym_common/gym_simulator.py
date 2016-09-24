"""This code is the simulator class for breakout game.
"""
import logging
from collections import deque
from functools import reduce
import time
import numpy

import bonsai

log = logging.getLogger(__name__)

INFINITY = -1
RECORDING_TIME = 40*60*60


class GymSimulator(bonsai.Simulator):

    CLASSIFIER = 0
    ESTIMATOR = 1

    def __init__(self, env, skip_frame=1, record_path=None,
                 random_seed=None, interface=None, render_env=True):
        """The constructor: Initialize the simulator parameters.
        """
        bonsai.Simulator.__init__(self)
        # Simulator parameters.all.
        self.env = env
        self.DEQUE_SIZE = 1
        # TODO: Make this into a dictionary to conform to the new API
        self._state_deque = deque(maxlen=self.DEQUE_SIZE)
        self._terminal = False
        self._episode_number = 0
        self._frame_count = 0
        self._reward = 0
        self.EPISODE_LENGTH = INFINITY
        self._start_time = time.time()
        self._is_recording = True if record_path else False
        self._skip_frame = skip_frame
        self.interface = self.CLASSIFIER if interface is None else interface
        self.gym_total_reward = 0.0
        self._render_env = render_env
        # TODO: Support multiple actuators by setting the size of
        # the ndarray with the inkling/server message
        self._action = (0 if self.interface == self.CLASSIFIER else
                        numpy.asarray([0]))
        self.env.seed(random_seed)

        self.env.reset()
        if self._is_recording:
            self.env.monitor.start(record_path)

    def _append_state(self, current_state):
        """This method appends the current state to the state deque.
        """
        self._state_deque.append(current_state)
        # Append the same frames into the deque if the real data size
        # in deque is less than the capacity of deque.
        while len(self._state_deque) < self._state_deque.maxlen:
            self._state_deque.append(current_state)

    def _reset_state(self):
        """This method resets all state related parameters and adds
        the first observation.
        """
        self._state_deque.clear()
        self._episode_number += 1
        log.info("Epsiode %d is starting now...", self._episode_number)
        first_state = self.process_observation(self.env.reset())
        self._append_state(first_state)

    def set_properties(self, **kwargs):
        """Set the properties of gym simulation.
        """
        self.EPISODE_LENGTH = kwargs["episode_length"]
        self.set_deque_size(kwargs["deque_size"])

        log.info("Gym episode length set to: {}"
                 .format(self.EPISODE_LENGTH))
        log.info("Gym deque size set to: {}"
                 .format(self.DEQUE_SIZE))

        self._reset_state()

        return True

    def set_deque_size(self, size):
        """
        Sets the active memory length of the observer.
        """
        self.DEQUE_SIZE = size
        old_deque = self._state_deque
        self._state_deque = deque(maxlen=self.DEQUE_SIZE)

        states = []
        while len(old_deque) > 0:
            states.append(old_deque.popleft())

        for state in states:
            self._append_state(state)

    def _check_terminal(self, done):
        """
        Checks if a Gym episode has completed.
        """
        if (done or
            (self._frame_count > self.EPISODE_LENGTH and
             self.EPISODE_LENGTH != INFINITY)):
            print("Episode {} reward is {}".format(
                self._episode_number, self.gym_total_reward))
            self.gym_total_reward = 0.0

            self._reset_state()
            self._terminal = True
        else:
            self._terminal = False

    def process_observation(self, obvs):
        """
        Calculates the luminance of the values for the image and
        performs nice preprocessing.
        """

        return obvs

    def get_state_schema(self, state):
        # This is tied with T214
        return state

    def set_prediction(self, **kwargs):
        self._action = (kwargs['action'] if
                        self.interface == self.CLASSIFIER else
                        numpy.asarray([kwargs['action']]))

    def get_state(self):
        """
        This method should apply the input action to advance the game
        to the next state, and then it should return that state and the
        reward from advancing to that state. It should also return a
        boolean is_terminal, indicating if the game ended (this could happen
        if all the bricks are broken, or if we run out of lives, or if
        should_stop was specfiied as true).

        If the input should_stop is true, the game should stop, and it
        should re-initialize itself to the starting state based on the
        hardcoded properties.

        If the game itself is terminal (for example you ran out of lives,
        or you broke all the bricks), reset the game as well as ai settings.
        """
        # Step 1: Perform the action and update the game along with
        # the reward.
        average_reward = 0
        for i in range(self._skip_frame):
            observation, reward, done, info = self.env.step(self._action)
            self._frame_count += 1
            average_reward += reward
            self.gym_total_reward += reward

            # Step 2: Render the game.
            if self._render_env:
                self.env.render()

            if done:
                break
        self._reward = average_reward / (i + 1)

        time_from_start = time.time() - self._start_time
        if self._is_recording and (time_from_start > RECORDING_TIME):
            self.env.monitor.close()

        # Step 3: Get the current frames, and append it to deque to get current
        # state.
        current_frame = self.process_observation(observation)
        self._append_state(current_frame)

        # We append all of the states in the deque by adding the rows.
        current_state = reduce(
            lambda accum, state: accum + state, self._state_deque)

        # Step 4: Check if we shousld reset
        self._check_terminal(done)

        # Step 6: Convert the observation to an inkling schema.
        return self.get_state_schema(current_state)

    def open_ai_gym_default_objective(self):
        return self._reward

    def get_terminal(self):
        return self._terminal
