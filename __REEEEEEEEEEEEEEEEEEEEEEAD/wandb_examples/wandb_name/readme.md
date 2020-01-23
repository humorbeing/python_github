### my exp
- command is in venv bin
    - `~/Desktop/Link to Mystuff/Workspace/python_world/Venv/3.5_pytorch1.3/bin/wandb`
- login
    - login key might be same everytime
    - `~/Desktop/Link to Mystuff/Workspace/python_world/Venv/3.5_pytorch1.3/bin/wandb login xxx`
    - key is `a6f5079f5d5476735d22bac595bb76c5aa1cb369`
    - this will creat wandb folder inside this folder
        - go to runner code folder, and execute the command
        - will it designate a project? can i change it later in init?
            - it DID NOT specify a project to upload
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

- `print('anything')` is being recorded and can check at website

- specify project to upload
```
import wandb
wandb.init(project="xxxx")
```
    - xxxx is the project name