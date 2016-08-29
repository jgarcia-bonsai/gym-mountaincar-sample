import unittest
import bonsai
import bonsai.gym
from bonsai.gym import GymSimulator


# We only test the gym simulator if we have
# gym installed on the system.
try:
    import gym

    class TestGymSimulator(unittest.TestCase):

        def setUp(self):
            env = gym.make('MsPacman-v0')
            self.gymsim = GymSimulator(env)

        def test_append_state(self):
            self.gymsim._append_state(1)
            assert (len(self.gymsim._state_deque)
                    is self.gymsim.DEQUE_SIZE)

        def test_set_deque_size(self):
            self.gymsim._append_state(1)
            self.gymsim._append_state(2)
            self.gymsim._append_state(3)

            # Upsizing
            prev = len(self.gymsim._state_deque)
            self.gymsim.set_deque_size(10)
            assert (len(self.gymsim._state_deque)
                    is 10)
            assert (self.gymsim.DEQUE_SIZE
                    is 10)

            # Downsizing
            self.gymsim.set_deque_size(2)
            assert (len(self.gymsim._state_deque)
                    is 2)
            assert (self.gymsim.DEQUE_SIZE
                    is 2)

            assert self.gymsim._state_deque[-1] is 3

        def test_set_properties(self):
            self.gymsim.set_properties(episode_length=10,
                                       deque_size=4)

            assert (len(self.gymsim._state_deque)
                    is 4)
            assert (self.gymsim.DEQUE_SIZE
                    is 4)
            assert (self.gymsim.EPISODE_LENGTH
                    is 10)

    if __name__ == '__main__':
        unittest.main()
except ImportError:
    pass
