# abbreviation dictionary
- WM: world model
- CL: competing loss
- DMB: Discriminative Model Based RL
    - competing loss is inaccurate term.
    from now on, use discriminative loss.
    So is DMB
- MB: model based method, means I train with
conventional model based method.

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
    - [x] Train a simple RNN model.
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
    - [x] make competing loss model-learning model.
        - so, loss is going inif negative, I will try exp() on it.
        hope math works.
        - problem. loss is not diverging. maybe need more data,
        maybe need more layers.
            - [x] better log
            - [x] more data  <---
            - [x] more layer
    - [ ] make a normal one
    - [ ] adverserial one


# Working On (Top is latest)

- [x] conduct experiment
- [x] transfer encodr weight to a3c platform
- [x] add a max ceiling on maximizing loss
- [x] found a bug, when logging, lss should be np.mean(), not loss.
- [x] make a normal encoder decoder
- [x] save model function (good news on lambda 0.85)
- [x] more layers
- [x] (competing loss)test lambda which give penalty to maximizing loss
- [x] cuda train
- [x] Better logging
- [x] More data on competing loss WM
- [x] Checking math code on competing loss WorldModel

# online train CL model
after cl model is converged, find a way to online train it with
policy gradient. it looks like cl model can grab good env information
at first, but those information is not that valuable for later state.
it can mean that later state is unseen state, so it needs to train on later state
as well.

# make training CL model schedule
first is 0.01, and max limit is 20, then 0.001, max limit is 50, etc.

# in pretrained weight with decoder on st
since i have pretrained weight on Competing loss model. I can use it to train 
decoder to state1 as well.

# competing loss lambda find.
Looks like the loss is not separating as I hoped.
if maximizing is wining the battle, minimizing will compromise
its goal, vise versa. why? maybe model is not complex enough.
- due to loss explosion, how about add a max number it can get.
so, add a max loss on maximizing loss, give minimizing more benefit.

# multiple model train encoder
- encoder decoder
- state1 encoder + state2 encoder to action
- state1 encoder + action to z to decoder state2
- state1 encoder + rnn to action correlate z2 to decoder to stat2s competing loss
- competing loss to state1 state2 action

# Not Competing loss, train a action preditor
basicly, putting a s_t s_t+1_hat --> predict actions.
1. use states for everything
2. use latent value for everything

before that, should I try why competing loss doesn't work.
or make normal model based first?

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


