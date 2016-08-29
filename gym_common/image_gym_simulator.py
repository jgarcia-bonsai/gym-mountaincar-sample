import bonsai
from bonsai.inkling_types import Luminance
from gym_common import GymSimulator

IMAGE_DEQUE_SIZE = 4


class ImageGymSimulator(GymSimulator):

    """Handles openAI gyms with raw pixel data."""

    def __init__(
            self, env, record_path, width, height,
            downsample, render_env=True):
        super().__init__(
            env, skip_frame=4, record_path=record_path, render_env=render_env)
        self.set_deque_size(IMAGE_DEQUE_SIZE)
        self.width = width
        self.height = height
        self.downsample = downsample

    def process_observation(self, obvs):
        """
        Calculates the luminance of the values for the image and
        performs nice preprocessing.
        """

        R = obvs[:, :, 0]
        G = obvs[:, :, 1]
        B = obvs[:, :, 2]

        # Calculates weighted apparent brightness values
        # according to https://en.wikipedia.org/wiki/Relative_luminance
        obvs = 0.2126 * R + 0.7152 * G + 0.0722 * B

        # Normalize the observations.
        obvs /= 255.0

        # should be replaced with tranformd
        obvs = obvs[::self.downsample, ::self.downsample]

        return obvs.ravel().tolist()

    def get_state(self):
        state = super().get_state()
        return {"observation": state}

    def get_state_schema(self, state):
        return Luminance(
            int(self.width / self.downsample),
            int(self.height * self.DEQUE_SIZE / self.downsample), state)
