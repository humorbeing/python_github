# To Do
- [x] Code Awesomisation
    - [x] 4 expriments code
- [x] encoder vs vae vs no train
    - [x] dataset generator
    - [x] vae
    - [x] encoder
    - [x] un trained vae and encoder?
- [x] extreme rnn
- [x] Small, Simple Domain
- [ ] awesomisation
    - [x] unify model
    - [Not Working] unify input x
- [x] 3 moves with small simple domain
- [ ] test with max possibility.
- [ ] different seed.
# More RNN?
after review the plot, Maybe all RNN?

# Need small and simpler environment
- use reward more frequently.
- Result is not ideal. Maybe testing is not correct?
# NEW in old_code
In old_code folder, NEW is the latest(newest)
version of commonly used code.

# gym episode max limit
 
```text
env = gym.make('Pong-v0')
env._max_episode_steps = 5
```
Change `5` to 100000 ish, to complete training.
On rnn experiment, agent can't learn further 
because of the limit is 10000.
# model input and output
- idea: input is raw pixel, not preprocessed pytorch tensors
    - [x] make a distinct difference between frame work
- idea: output is, state value V and action prob (this moment)
    - state value is numpy? or pytorch?
    - action prob, or should i add real action
    - [x] since above's are only output I am generating. RNN should work inside.
        - only action output is not working. Turns out, when training an RNN in multiprocess setup, you need to pass hx, cx from outside.
    - how about, since it's a policy, only output action, that can use directly.


# Small domain Pong-ram-v0
- idea: save time, more observation.
- training seems working but not fast enough
- since it's only MLP
    - RNN?
    - pixel - pixel?
        - s1+ 255 -s2 / 255*2
- experiment idea
    - only MLP
    - RNN
    - pixel - pixel
    - RNN + pixel

# Idea
- (CNN + LSTM + reconstruction) * n
- Attention Play game
- Half transfer learning
- lstm for critic and actor


