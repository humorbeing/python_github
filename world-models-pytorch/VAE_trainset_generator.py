
import numpy as np
from gym.envs.box2d import CarRacing
from to_import import _process_frame
from to_import import forward_favor_action
from to_import import ROOT_PATH


MAX_GAME_TIME = 1000
MAX_RUNS = 20
name_this = 'rollout_v0'
dst = ROOT_PATH + 'VAE_trainset/'


def VAE_trainset_generator(
        action_function,
        dst,
        name_this = 'rollout_v0',
        MAX_GAME_TIME = 1000,
        MAX_RUNS = 20,
        on=0,
        is_render=False,
        is_vebo=False
):
    env = CarRacing()
    states = []
    actions = []
    for run in range(MAX_RUNS):
        env.seed(seed=5)
        state = env.reset()
        env.render()  # must have!
        for game_time in range(MAX_GAME_TIME):
            if is_render:
                env.render()
            action = action_function(state)
            state = _process_frame(state)
            states.append(state)
            actions.append(action)
            state, r, done, _ = env.step(action)
            if is_vebo:
                print(
                    'RUN:{},GT:{},DATA:{}'.format(
                        run, game_time, len(states)
                    )
                )
        env.close()
    states = np.array(states, dtype=np.uint8)
    actions = np.array(actions, dtype=np.float16)
    save_name = name_this + '_{}.npz'.format(on)
    print('saved: ' + save_name +' len:', len(states))
    np.savez_compressed(dst + '/' + save_name, action=actions, state=states)


if __name__ == '__main__':
    VAE_trainset_generator(
        forward_favor_action,
        dst,
        name_this=name_this,
        MAX_GAME_TIME=MAX_GAME_TIME,
        MAX_RUNS=MAX_RUNS,
        # is_render=True
    )