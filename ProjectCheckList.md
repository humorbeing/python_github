### Check List
- [ ] level one is divided into different computers.
    - for git update
    - `/project/computerA`
- [ ] level two is runnable code versions.
    - for visit anytime
    - same for colab
    - `/project/computerA/code_v0001`
- [ ] **`args`** as controlling parameter of the project
- [ ] **`args`** logs into csv list when run
- [ ] print this **`args`** when run
- [ ] when logging, need a save name naming policy from **`args`**
    - model save name
    - log save name
- [ ] modularize project
    - [ ] **`args`**
    - [ ] runner(**`args`**)
    - [ ] log utility
        - naming policy
    - [ ] utility
    - [ ] models
- [ ] **`args`** parameter naming policy
    - args.log_path
    - args.model_save_path
    - args.data_path
- [ ] log has `all_logs`, `received_logs`, `saved_logs`, `sent_logs`
    - `all_logs`, `received_logs`, `saved_logs` is in main log folder
    - `sent_logs` is in sub log file (other computers, colab)
        1. logA is sent by computer A and moved into `sent_logs` in computer A
        1. logA is in `received_logs`
        1. logA is merged into `all_logs`
        1. logA is moved into `saved_logs`
- [ ] program one args at a time
    - full project structure, have quickrun
    - one args at a time and test it
        - test it separately if local
        - test it fully if global
- [ ] experiment record
    - [x] [collected] [running PC] experiment info
    - idea: `experiment info`
    - deployed: `[running PC] experiment info`
    - log collected: `[collected] [running PC] experiment info`
    - end: `[x] [collected] [running PC] experiment info`
- Example of folders
    - readme.md
    - Main PC:
        - `/project`
            - `/code_v0001`
                - `/logs`
                - `/sent_logs`
            - `/code_v0002`
                - `/logs`
                - `/sent_logs`
            - `/gather_logs_here`
                - `/all_logs`
                - `/received_logs`
                - `/saved_logs`
            - `/old_code`
                - `/old_code_from_A`
                    - `/logs`
                    - `/sent_logs`
    - Computer A:
        - `/project`
            - `/code_v0001`
                - `/logs`
                - `/sent_logs`
            - `/code_v0002`
                - `/logs`
                - `/sent_logs`

### version
- version is upgrade of running structure.
adding more args, fixing bugs.
- adding more models is not one of them.
as long as I can control with existing args,
that is not a change of versions.
- once the version is saturated to certain
stable program, burst out experiments.

### work directory / folder
- where project load data, save models, save demos
- work directory as root
    - /data has datasets
    - /save_xxx to save somethings
- path should have root come in, assemble children path in the project
- this root is outside of github monitoring.
    - it should be un-limited by github upload, download speed.
    - github updates should've free from this burden.
    - logs-like small text is ok with github monitoring.