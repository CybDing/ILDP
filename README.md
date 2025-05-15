# ILDP 2025 project (test only, build with interest)
Special thx to:
Real Stanford: diffusion policy (# main reference for transition)
the new genesis version for imitation learning using DP!

Structure ( Expected:( )
- env
  - flexiv_env: Providing basic settings with flexiv robotic arm(rizon 4)
- model
  - diffusion
    - diffusion_unet (To be updated soon!)
    - conv1d (To be updated soon!)
  - RL
    - (soft?)actor-critic
    - (... Benchmarks)
  - VLA
    - (Generative policy based on decoder-encoder transformer)
    - ...
  - VPP (To be estimated)
- env_runner
  - pushT_envRunner
  - ... 
- datasets
  - pushTDataset (To be updated soon!)
  - ...
- 
- utils
  - ...
- dataCollection
  - An interactive interface using Pygame communicating with genesis for user-friendly robot control(To be planned and executed soon!)   
- Workspace
  - train
  - eval
