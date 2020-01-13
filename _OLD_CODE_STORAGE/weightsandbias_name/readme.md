### my exp
- command is in venv bin
    - `/mnt/D8442D91442D7382/Mystuff/Workspace/python_world/Venv/3.5/bin/wandb`
- login
    - login key might be same everytime
    - `/mnt/D8442D91442D7382/Mystuff/Workspace/python_world/Venv/3.5/bin/wandb login xxx`
    - `/mnt/D8442D91442D7382/Mystuff/Workspace/python_world/Venv/3.5/bin/wandb login a6f5079f5d5476735d22bac595bb76c5aa1cb369`
    - key is `a6f5079f5d5476735d22bac595bb76c5aa1cb369`
- recording model graph
    - have tried various ways to trigger model saving
    - only triggers when there is backward() to model
    ```python
    import wandb
    wandb.init()
    model = M()
    wandb.watch(model)    
    y = model(x)
    loss = torch.mean(y)
    loss.backward()
    ```
- change which project upload to
    - \wandb\settings
    - change the `project = xxxx`
    - into wanted prject (get it from website)
    - in this example, `tt` is uploading to upper level of project.