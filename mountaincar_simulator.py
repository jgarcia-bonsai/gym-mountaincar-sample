import gym

import bonsai
from bonsai_gym_common import GymSimulator

ENVIRONMENT = 'MountainCar-v0'
RECORD_PATH = None
SKIPPED_FRAME = 4


class MountainCarSimulator(GymSimulator):

    def __init__(self, env, skip_frame, record_path, render_env):
        GymSimulator.__init__(
            self, env, skip_frame=skip_frame,
            record_path=record_path, render_env=render_env)

    def get_state(self):
        parent_state = GymSimulator.get_state(self)
        state_dict = {"x_position": parent_state.state[0],
                      "x_velocity": parent_state.state[1]}
        return bonsai.simulator.SimState(state_dict, parent_state.is_terminal)


if __name__ == "__main__":
    env = gym.make(ENVIRONMENT)
    base_args = bonsai.parse_base_arguments()
    simulator = MountainCarSimulator(
        env, SKIPPED_FRAME, RECORD_PATH, not base_args.headless)
    bonsai.run_with_url("mountaincar_simulator", simulator,
                        base_args.brain_url)
