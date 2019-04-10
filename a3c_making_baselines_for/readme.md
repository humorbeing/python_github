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
- [x] test with max possibility.
- [x] different seed.
- [ ] encoder model based model
    - [ ] Train a simple RNN model.
        - Don't know how LSTM vs LSTMCell. And I need a simple version of
        RNN training sessions to use in further coding. E.g. new pytorch test code.
    - [ ] train encoder from scratch
        - train with raw sequence
    - [ ] train encoder first then model
        - train with extracted sequence
- [ ] encoder with gradient through the network
- [ ] Understand RL
    - [ ] a2c
    - [ ] ppo
    - [ ] trpo
    - [ ] dqn
- [ ] make model-based architecture.
    - [ ] make competing loss model-learning model.
        - so, loss is going inif negative, I will try exp() on it.
        hope math works.

# Idea on model learning with competing loss
rnn model learning network doesn't take in action vectors,
instead, it generates num_action x latent vector, then decoded
next state. And loss is like with corresponding action predicted state,
the reconstruction loss is getting minimized; tho, the other action predicted
states are getting loss maximized. So trying to squeeze network to
learn the important part of action related features.
 - another thought: only use encoder, and loss is calculated using latent vectors.

# Train a simple RNN model
- [x] make identity matrix
- [ ] lstmcell can't be trained with batches
- [ ] use lstm in pytorch
    - It looks like I might have made
    a mistake in my early implementation
    of 'World models' with LSTM part.
    I think I need to study pytroch world models
    implementation codes? the problem in these code
    is, when you are training lstm with one batch of 1000
    sequence, How should i manage hidden and cell layers
    inputs. and in the 'internet code', there seems to be
    a LSTM train and LSTMcell use kind of structure.
    About the my mistake part, I think I screwed up training
    part. 



# Gradient injection
- encoder is trained with 
    - decoder error
    - model based error
        - with or without decoder
    - policy gradient error
    - inverse model error
    - state action to previous action error
    - any combination of above
        - IDEA: encoder with
             - policy gradient
             - model based with decoder
             - inverse model
             - SA pre S model

# About pixel - pixel
Why didn't id work at all. I think this
128 vectors of representation doesn't work
like image pixels.
So, what it means is, it's not really
`pixel - pixel`, more like,
`state - state` and the result is not 
obtaining much information about the
game state MOTIONS.

# argmax on action selection
- turns out this yields good result, I am going to use this as new metric.



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


