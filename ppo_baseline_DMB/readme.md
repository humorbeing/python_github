# What I did
- dissect pytorch-ppo
    - Problem: reward was not -1 on pong env.
    It is caused by envs.py normalization funtion.
    I thought this function is only normalizing obs space.
    But, it also changes rewards.
    - What happened here is standardization of input data



# Old: down below [update at: 2019/7/22]
- [ ] dissect ppo code
    - [x] ppo simple
    - [x] move37
    - [x] ppo pytorch
    - [ ] baselines
- [ ] my own ppo from pytorch

