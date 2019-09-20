# Exp run v7
- v0002: pong D v4 3 actions, with gamma, gae, value loss clip, adv stdiz,
GPU with RNN
- expri_run_v0002.py
- log: ppo_PongD43act_64RNN_gpu

# Exp run v7
- v0002: pong D v4 3 actions, with gamma, gae, value loss clip, adv stdiz,
GPU with RNN
- expri_run_v0001.py
- log: ppo_PongD43act_RNN_gpu


# Plan v7
- [x] rnn
- [x] rnn + gpu
- [ ] need to fix non-rnn
- [ ] need to fix non-rnn gpu

# Exp run
- v0002: pong D v4 3 actions, with gamma, gae, value loss clip, adv stdiz,
GPU
- exp_run_v0002.py
- log: ppo_PongD43act_ret_gamma_gae_adv_stdiz_value_loss_clip_gpu

# Exp run
- v0001: pong D v4 6 actions, with gamma
- Exp_run_v0001.py
- log: ppo_PongrD4_6act_gamma

# Idea v6
- a2c, strip all ppo stuff.
- NoFrameskip of vector version.
- number of mini batches when after rollouts and updates.
the number should be big to be IID sampling.

# Plan v6
- [x] training epoch limit
    - ATM, it's 1e7
    - Nope, set to 5e6, for quick eval
    - Need a forever training?
- [ ] exp data plot
    - [x] evaluation plot
    - [ ] training plot
- [x] gpu
    - [x] seeded pytorch
- [x] gae
- [x] advantage standardization
- [x] value loss clip
- [x] Env gamma return parameter

# Experiment run 
- log name: ppo_PongrD4_3act_withgamma_popvar
- env: Pong-ramDeterministic-v4
- with gamma standardization
- standardization is population variance
- 3 actions

# Experiment run (same, continue)
- log name: ppo_PongrD4_3act_nogamma_popvar_evlsame
- env: Pong-ramDeterministic-v4
- no gamma standardization (only obs, no rewards)
- standardization is population variance
- 3 actions

# Evaluation might be doing the same evn forever.
- evaluation is always rounded number like 18.000, 21.000.
this means all evns are same. need to fix the random seed.
- Big problem. env.seed() seems doesnt affect how game is progressed.
It means, every time the game is the same. At least for pong.

# Experiment run (Stopped due to evaluation might be)
- log name: ppo_PongrD4_3act_nogamma_popvar
- env: Pong-ramDeterministic-v4
- no gamma standardization (only obs, no rewards)
- standardization is population variance
- 3 actions

# v5 Plan
- [x] 3 action space
- [x] check evaluation process
    - looks fine atm.
- [x] training seems have batch problem. Whereas number of
mini batchs and total trainable samples are not exactly match.

# V5
- Pong-ramDeterministic-v4
- no gamma standardization 
- standardization is population variance
- doesn't train at all.

----------------------------------
# Experiment RUN
- log name: Pong-D-4-nogamma-nopopvariance
- env: Pong-ramDeterministic-v4
- no gamma standardization (only obs, no rewards)
- standardization is population variance

# Plan
- [ ] input data processing
    - [ ] re implement vec_normali.
        - [x] without gamma for return, i guess
            - so, there is an complication about calculating returns in vecnorm
            implementation from openai's baselines.
            - meanwhile, if i don't do gamma. then it seems only obs are being normalized.
            - [x] evaluation without gamma
            - [ ] in running mean std: experiment on sample variance vs population variance
        
- [ ] input data processing in evaluation
- [ ] v0 vs deterministic v4 experiment

# Done in Last version (v3)
- [x] logging system, from a3c
- [x] evaluation, learn from pytoch ppo code
    - I was logging training errors from last 10 episodes
    - in the code, there is a evaluation code, learning it.    


# v2 is deserted (given up) version.
- given up on pytorch free env
# This is most simple version of pytorch PPO
- No RNN
- No GAE
- No V function fancy error
- No Standardization
    - No input
    - No advantage value
- No Pixel
- No 4 frames
- No fancy logging system
- No GPU
- No fixing pytorch env data stream