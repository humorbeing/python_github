# Understanding the difference in envs from gym
1. `v0` vs `v4`
    - In `v0`, actions are stochastic, which has 25% to execute 
    previous action.
    - In `v4`, actions are what we are intended to execute
1. With or Without `Deterministic`
    - Without. The frame is skiped from 2 to 5. So it's a 
    stochastic frame skipping envs
    - With. The frame is skipped as in fixed number 4.
1. With NoFrameskip
    - with `Noframeskip`, there is no frame skipping from the 
    envs.
# So
- Pong-ram-v0 would have random frame skipping and 25% of previous
action executed.
- Pong-ramDeterministic-v4 would have fixed frame skipping and
always do the actions we intended to
- Pong-ram-v4 would have executed the actions we intended to, but because
of the random frame skipping settings, actions will be still have some
random portion in it.