import gym

import bonsai
from bonsai_gym_common import GymSimulator

ENVIRONMENT = 'MountainCar-v0'
RECORD_PATH = None
SKIPPED_FRAME = 4


class MountainCarSimulator(GymSimulator):

    def __init__(self, env, skip_frame, record_path, render_env):
        super().__init__(
            env, skip_frame=skip_frame,
            record_path=record_path, render_env=render_env)

    @property
    def get_state(self):
        state_schema = super().get_state
        return {"x_position": state_schema[0],
                "x_velocity": state_schema[1]}

if __name__ == "__main__":
    env = gym.make(ENVIRONMENT)
    base_args = bonsai.parse_base_arguments()
    simulator = MountainCarSimulator(
        env, SKIPPED_FRAME, RECORD_PATH, not base_args.headless)
    assert isinstance(base_args, object)
    bonsai.run_with_url("mountaincar_simulator", simulator,
                        base_args.brain_url)
