# DPPO 微调链路代码审计报告

> 日期：2026-06-12
> 范围：`genesis_ILDP` 中 PPO/DPPO 微调相关代码（collector → buffer → GAE → logprob → PPO update）
> 目标：今天跑通整条链路；核心算法（logprob / PPO loss / GAE 细节）由作者本人实现，本报告只标注留白处与需要思考的问题。

---

## 0. 总体结论

仓库里实际上存在**两套并行、都未完成**的 DPPO 实现：

| 路线 | 入口/组成 | 状态 |
|---|---|---|
| **A. Workspace 路线（推荐）** | `workspace/finetune_diffusion_rl_workspace.py`（自带 ReplayBuffer、GAE、训练循环） | 主循环逻辑完整，但引用的 `Critic` / `PPODiffusion` / `TrajsCollector` 三个类**都不存在**（名字或文件对不上） |
| B. Trainer 路线（建议先搁置） | `agent/trainer/base_trainer.py` + `finetune_ppo_trainer.py` + `rl_agent/base_agent.py` + `ppo_agent.py` | 半成品：`base_trainer` 没保存大部分构造参数、`_get_train_data` 末尾 TODO、buffer 字段与 runner 输出对不上 |

**今天要跑通，最短路径是只修 A 路线**，把 B 路线当作设计草稿/legacy。下面的 P0 清单按链路顺序排列，全部修完后链路可以空转（loss 数值还需要你自己写的核心算法才正确）。

---

## 1. P0 阻断性 Bug（不修就 import 失败 / 启动即崩）

### 1.1 import 阶段直接崩溃：非法的 `Dict[...]` 类型注解
- `genesis_ILDP/model/rl/critic.py:41` — `Dict[List|int]`
- `genesis_ILDP/model/rl/critic.py:64` — `Dict[int|list]`
- `genesis_ILDP/agent/rl_agent/base_agent.py:15` — `Dict[float|str]`

`typing.Dict` 必须给两个参数（已实测 `TypeError: Too few arguments for typing.Dict`），且函数签名注解在**类定义时即求值**，所以 `import critic / base_agent` 就崩，连带 `ppo_agent.py`、`finetune_ppo_trainer.py`、finetune workspace 全部不能 import。改成 `Dict[str, Any]` 或干脆删掉注解。

### 1.2 workspace 引用的 `Critic` 类不存在（其实在 backup 文件里）
- `finetune_diffusion_rl_workspace.py:35` 导入 `from genesis_ILDP.model.rl.critic import Critic`，但 `critic.py` 里只有 `BaseCritic / CriticObs / CriticImageObs`，没有 workspace 需要的接口（`shape_meta`、`layers_Sdim`、`adapt_obs_encoder()`、`.Qnet`、`.normalizer`）。
- **完整实现就在 `model/rl/critic-backup.py`**（构造签名、`adapt_obs_encoder`、`_create_QNet`、`forward(obs_dict)` 与 workspace 调用完全吻合），但文件名带连字符无法 import。
- **修法**：把 `critic-backup.py` 的 `Critic` 类并回 `critic.py`（或重命名为合法模块名）。注意它 import 的是 `diffusion_policy.model.vision.crop_randomizer`，应改为仓库内的 `genesis_ILDP.model.vision.crop_randomizer`（见 §4.1 子模块问题）。

### 1.3 `PPODiffusion` 类不存在
- `finetune_diffusion_rl_workspace.py:36` 导入 `PPODiffusion`；`ppo_agent.py` 里只有 `PPODiffusionAgent`，且：
  - 构造参数名不一致：workspace 传 `clip_vloss=`，agent 收 `clip_vloss_value=`；
  - workspace 调用的 `load_policy(policy)` 和 `loss(...)` 方法都不存在（只有 `_loss`，签名/语义不同）；
  - `PPODiffusionAgent.__init__` **没有调用 `super().__init__`**，导致 `self.actor / self.critic / p_optimizer / v_optimizer` 全部不存在，`_loss` 里 `self.actor.get_logprob(...)`、`self.critic(obs)` 必然 AttributeError；
  - workspace 从未把 critic 交给 ppo_trainer，但 `_loss` 内部要算 `new_value = self.critic(obs)` —— 需要补一个 `load_critic()`（或把 critic 作为构造参数）。
- **修法**：新建/重命名一个 `PPODiffusion` 类，按 workspace 的调用约定提供 `__init__(max_FTdiff_idx, clip_base_ploss, clip_rate_ploss, clip_vloss, clip_advantage_*_quantile, advantage_decay_coeff)`、`load_policy()`、`load_critic()`、`loss(chain_before, chain_next, advantage, obs, denoising_idx, old_value, old_logprob) -> (ploss, vloss)`。loss 内部数学是你要自己写的核心算法（见 §3）。

### 1.4 `TrajsCollector` vs `TrajCollector` 类名不一致
- `finetune_diffusion_rl_workspace.py:33` 导入 `TrajsCollector`；`agent/collector/traj_collector.py:9` 定义的是 `TrajCollector`。统一即可。

### 1.5 RL 微调配置文件缺失
- workspace 的 hydra 入口（`finetune_diffusion_rl_workspace.py:599-602`）指定 `config_name="finetune_diffusion_rl"`，但 `config/` 下根本没有这个 yaml。
- 需要新建 `config/finetune_diffusion_rl.yaml`，至少包含 workspace 引用到的全部字段：
  `checkpoint_path`、`critic.{layers_Sdim, n_obs, img_feature_dim, enable_crop, crop_shape, train_encoder}`、
  `ppo.{max_FTdiff_idx, clip_base_ploss, clip_rate_ploss, clip_vloss, clip_advantage_upper_quantile, clip_advantage_lower_quantile, advantage_decay_coeff}`、
  `optimizer.{policy_lr, critic_lr}`、
  `training.{device, num_epochs, batch_size, num_ppo_updates_per_epoch, n_rollout_episodes, max_rollout_steps, rollout_seed, gamma, gae_lambda, checkpoint_every, max_replay_episodes, max_grad_norm}`、
  `logging.*`。
- 顺带：现有 `config/finetune/finetune_action_diffusion_pusht_image.yaml` 里 `policy._target_` 写的是旧路径 `genesis_ILDP.policy.action_diffusion_image_policy.ActionDiffusionImagePolicy`（实际在 `policy/pretrain/` 下），该文件属于"换数据集监督微调"路线，与 DPPO 无关但同样是坏的。

### 1.6 `FTActionDiffusionImagePolicy.__init__` 与父类签名完全脱节
- `policy/finetune/finetune_action_diffusion_image_policy.py:16-38`：子类按**旧版**父类签名写的（`vision_backbone / scheduler_mode / crop_shape / img_encoding_dim ...`），而现在的父类 `ActionDiffusionImagePolicy.__init__`（`policy/pretrain/action_diffusion_image_policy.py:25`）要求传**已实例化的组件**：`cropper, pos_encoding, time_encoding, imgs_encoding_net, conditioned_unet, noise_scheduler`。
- 现状是 `super().__init__(shape_meta, normalizer, vision_backbone, diff_steps, ...)` 把 `vision_backbone` 喂给 `cropper`、`diff_steps` 喂给 `pos_encoding`……实例化必崩。
- **修法（最简）**：FT 子类 `__init__` 直接 `def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)`，让 hydra 用 checkpoint 里的 `cfg.policy`（含嵌套 `_target_` 组件）原样实例化。

### 1.7 FT policy 实例化路径写错
- `finetune_diffusion_rl_workspace.py:221`：`_target_` 设为 `genesis_ILDP.policy.FT_action_diffusion_image_policy.FTActionDiffusionImagePolicy`，实际模块是 `genesis_ILDP.policy.finetune.finetune_action_diffusion_image_policy`。

### 1.8 TrajCollector 一连串接口错误（`agent/collector/traj_collector.py`）
1. workspace `_create_trajs_collector`（:284-302）只给了 `{checkpoint_path, device}` 的最小 eval_cfg，但 `EvalActionDiffusionImageWorkspace.__init__` 在 `visual_manual_eval_...workspace.py:62` 要写 `self.eval_cfg.env_runner.enable_render` —— 最小 cfg 没有 `env_runner` 节点，直接崩。
2. `traj_collector.py:31-40` 改的是 `checkpoint_cfg.env_runner.*`，但训练 yaml 里 runner 在 **`task.env_runner`** 下（`config/train/train_action_diffusion_pusht_image.yaml:142`）。
3. `traj_collector.py:37` 写 `max_step`，runner 构造参数叫 **`max_steps`**。
4. **`episode_recording` 从未置 True**：`PushTImageRunner` 默认 `episode_recording=False`（`pusht_image_runner.py:49`），训练 yaml 也没设。不开它，`predict_action(recording_diffusion=False)` 不会记录扩散链，`episode_buffer` 永远是空的——收集到的轨迹没有 action 链，整个 DPPO 没数据。
5. `traj_collector.py:45`：`self.env_runner.run()` **没传 policy**，而 `PushTImageRunner.run(self, policy, ...)` 第一个位置参数就是 policy。应传 `self.policy`（workspace 已经把 FT policy 挂到 `collector.policy` 上了）。

### 1.9 runner 录制的扩散链形状与缓冲区声明不符
- `pusht_image_runner.py:138/175`：`episode_buffer['action']` 预分配为 `(0, diff_steps+1, n_action_steps, 2)`，但 policy 记录的扩散链是整个 **horizon**（16），不是 `n_action_steps`（8）——`action_generation_ddpm` 的快照是 `predicted_trajs`（`(B, horizon, Da)`，见 `action_diffusion_image_policy.py:355/390`）。第一次 `torch.cat` 即形状不匹配。把缓冲区第三维改成 horizon（并想清楚你之后 logprob 要对整条 horizon 还是只对执行的 n_action_steps 段算，见 §3.G）。
- 另外 runner 的 `diff_steps` 默认 100 是独立参数，必须与 checkpoint 的 `diff_steps` 一致，建议从 cfg 透传而非靠默认值。

### 1.10 FT policy 内部小错（`policy/finetune/finetune_action_diffusion_image_policy.py`）
- `:41` `assert hasattr(FTActionDiffusionImagePolicy, 'normalizer')` —— 检查的是**类**属性，永远 False/无意义，应为 `self`。
- `:58` `self.normalizer['img']` —— normalizer 的 key 是 **`'image'`**（见 `set_normalizer` 的断言，`action_diffusion_image_policy.py:261`），KeyError。
- `:55` `image.transpose(dim0=1, dim1=-1)` 对 5D `(B,To,H,W,C)` 是把 To 和 C 换位，错误；应按父类 `_preprocess_obs` 用 `permute(0,1,4,2,3)`。
- `get_logprob` 里 `self._encoding_obs(image, agent_pos)` 默认 `training=True` → 推理 logprob 时用了**随机裁剪**，与采样时的 center-crop 不一致（会让 old/new logprob 各自带噪声）。应传 `training=False`，或者想清楚你要不要让裁剪随机性进入 ratio。
- `get_logprob` 喂 U-Net 前**没有转置**：父类约定输入 `(B, Da, T)`（`compute_loss` :334 与采样 :365 都先 `transpose(1,2)`），这里直接把 `(B, T, Da)` 喂进去了；输出同样要转回来。
- `self.time_encoding(denoising_idx)` 传的是 `(B,)` 张量，但 `time_encoding.forward`（`model/encoding.py:46-67`）只支持**标量**（内部 `torch.cat(..., dim=0)`）。父类的做法是逐元素 stack（`compute_loss:329`）。
- `self.betas[denoising_idx]` / `self.cum_alphas[denoising_idx-1]`：
  - `denoising_idx=0` 时 `-1` 索引回卷到最后一个元素，数值错误；
  - `betas/cum_alphas` 没有 `register_buffer`（`action_diffusion_image_policy.py:100-103` 注释掉了），`policy.to(device)` 不会搬它们；用 CUDA 索引张量去索引 CPU 张量会直接报错。预训练能跑通是因为 scheduler 在自己的 `device` 上建张量——微调时务必确认两者同设备（最好恢复 `register_buffer`）。

### 1.11 其余调用方的小不一致（如果以后启用 B 路线 / DDIM 才会踩）
- `ppo_agent.py:107-113` 调 `get_logprob(..., chain_after=...)`，而 FT 定义是 `chain_next` —— 关键字不匹配。
- `predict_action`（`action_diffusion_image_policy.py:222-236`）在 `use_ddpm=False` 时走 `action_generation_ddim`，它**只返回单个张量**，而 `recording_diffusion` 分支按 tuple 解包——用 DDIM 收集轨迹会崩（TODO 注释在 :231 也承认了）。今天先固定用 DDPM 收集。
- `finetune_ppo_trainer.py` / `base_trainer.py`：`base_trainer.__init__` 没保存 `env_collector/evaluator/batch_size/train_epochs/per_epoch_iterations`（子类用到 `self.per_epoch_iteration`、`self.batch_size`、`self.env_collector` 都会 AttributeError）；`_get_train_data` 引用的 `collect_trajs['chain_before'/'chain_after'/'logprob'/'length'/'truncate']` 在 runner 输出里都不存在；末尾 GAE 是 TODO。**建议整体标记 legacy，不要修。**

---

## 2. P1 数据流/一致性问题（能跑起来后第一批要核对的）

1. **obs 的 /255 一致性**：`predict_action` 在 normalize 前做了 `imgs / 255.0`（:199），而 FT `get_logprob` 的 `_process_obs` 没做。buffer 里存的是 env 原始 obs，两条路径对同一张图会得到不同归一化结果 → logprob 系统性偏差。统一处理（建议在 `_process_obs` 里复刻 predict_action 的预处理）。
2. **ReplayBuffer 的 `episode_ends` 类型混用**：runner 给的是 `torch.int64` 张量，workspace 的 ReplayBuffer 把它当 list `extend`（`finetune_diffusion_rl_workspace.py:95`），后续 `[0] + episode_ends[:-1]`（:348）恰好能跑，但元素是 0-d tensor，容易埋雷。建议入口处统一 `.tolist()`。
3. **`last_obs` 没接进来**：runner 辛苦存了 `last_obs`（用于截断 episode 的 bootstrap），ReplayBuffer 的 `add_trajectories` 直接丢弃，`is_truncate` 也没存 → GAE 把截断当终止（见 §3.F）。
4. **`n_test_vis=n_episodes`**（`traj_collector.py:34`）：收集数据时全开可视化录像，纯浪费时间，应为 0。
5. **eval workspace 的 EMA 加载键不一致**：eval 用 `payload['state_dicts']['ema_model']`，RL workspace `_load_and_convert_policy` 用 `payload['ema_model']`（:226）——确认你的 checkpoint 实际结构，统一一个。
6. `run_rl_finetune_dp.py` **名不副实**：它做的是"换数据集的监督微调"，且 argparse(`--checkpoint` required) 和 hydra 的命令行解析互相打架，基本不可用。DPPO 的真正入口是 `finetune_diffusion_rl_workspace.py` 的 `main`。建议改名或在 task/ 下新建一个干净的 `run_dppo_finetune.py` 薄入口。

---

## 3. 核心算法留白（按你的要求：列出问题，不替你写）

这些是"链路通了之后数值才正确"的部分，全部集中在你说要自己写的核心算法里：

- **A. 高斯 logprob 公式**（`finetune_..._policy.py:get_logprob`，:106-122）
  当前 `log_prob = -((chain_after - action_pred_mean) / variance)` 不是任何分布的 log density：
  - posterior 均值不是 `chain_before - noise_pred`，应该是 DDPM 后验均值 `μ = 1/√α_t · (x_t − β_t/√(1−ᾱ_t) · ε_θ)`（采样代码 `action_generation_ddpm:371-374` 里就是这么写的，可直接对照）；
  - 缺平方项、缺 ½、分母应是 σ²（且 `variance` 是方差，不是 std）、缺 `-½·log(2πσ²)` 项（常数项对 ratio 无影响，但对调试可读性有影响）；
  - 想清楚 logprob 是对 `(T, Da)` 每维独立求和（对角高斯），还是只对执行段求和（见 G）。
- **B. PPO ratio 与 clip 目标**（`ppo_agent.py:_clip_sigratio/_loss`）
  - ratio 应为 `exp(new_logprob − old_logprob)`，当前是 `new_logprob / old_logprob`（两个负数相除，方向都不对）；
  - 标准 PPO 目标是 `−min(r·A, clip(r,1−ε,1+ε)·A)`，当前只有 clipped 一项，丢了 unclipped 分支（clip 失去"悲观下界"含义）；
  - `advantage_decayed` 用 `torch.Tensor([...])` 在 CPU 上现造（:118），CUDA 下设备不匹配；用 `self.advantage_decay_coeff ** denoising_idx`（张量幂）即可顺带解决；
  - 你设计的"按去噪步衰减 clip 宽度 / advantage"是 DPPO 论文里 denoising-step-dependent 的变体，值得保留，但注意 `weight_advantage_ploss` 缓存后 `max_FTdiff_idx` 改变不会失效。
- **C. Critic 的回归目标丢了**
  - workspace `_compute_gae_advantages` 算出了 `returns`（:370）但只返回 `advantages, values`；
  - `_loss` 里的 value clip 写成了 `(value_clipped − old_value)²`（`ppo_agent.py:130`）——目标里没有 returns，**critic 在朝自己的旧输出回归，没有任何学习信号**。标准写法两支都应对 `returns` 求 MSE。
- **D. denoising index 的语义统一**（最容易出错的一处，建议先画清楚再写代码）
  - 录制链 `action_buffer` 索引 0 = 纯噪声（t = diff_steps），索引 K = 最终动作（t = 0）；
  - 而 `get_logprob` 里 `betas[denoising_idx]` 把同一个 idx 当 DDPM 的 t 用 —— 两套坐标系刚好**相反**，必须换算 `t = diff_steps − k`；
  - `max_FTdiff_idx` 现在采样 `[0, max_FTdiff_idx]` 取的是**高噪声端**；DPPO 论文微调的是**最后 K 步去噪**（低噪声端）。想清楚你要哪端，然后让 workspace 采样、clip 宽度衰减、advantage 衰减三处方向一致。
- **E. old_logprob 与训练时的 (t, x_t, x_{t-1}) 不配对**
  - `_compute_old_logprob_and_value`（workspace:377-441）随机采一组 denoising_idx 算 old_logprob；`_train_ppo_epoch`（:541-553）又**独立**随机采一组 idx 算 new_logprob —— 两者对应不同的转移，ratio 没有意义。
  - DPPO 标准做法：把每个 env step 的每个去噪转移当成一个"子 MDP 步"，old_logprob 按 (step, k) 成对存储；或者在训练采样时把 (idx, chain_before, chain_next, old_logprob) 一起采出来。这是你设计数据结构时要先定的事。
- **F. GAE 细节**
  - workspace 内置 GAE 在 episode 末尾一律 `next_value = 0`（:359-362），截断 episode 应该用 `last_obs` 的 V 值 bootstrap（数据 runner 已经存了，buffer 丢了，见 §2.3）；
  - ReplayBuffer 跨 epoch 保留旧轨迹（`max_episodes=1000`），但 PPO 是 on-policy 的——旧 epoch 的 chain 来自旧策略，old_logprob 每 epoch 重算只是掩盖了这一点。要么每 epoch 清 buffer，要么明确自己在做某种 off-policy 修正；
  - `rl_utils.py:calculate_GAE` 的 `is_truncate` 掩码语义与命名相反（乘 `is_truncate` 意味着只有截断才 bootstrap 后又把 done 也乘进了递推），且与 workspace 内置 GAE 重复——二选一，删掉另一个。
- **G. logprob 的作用范围**：扩散链记录的是整个 horizon=16 的动作序列，但 env 只执行了 `n_action_steps=8`（且从 `start = n_obs_steps−1` 开始，见 `predict_action:239-241`）。reward 只对应执行段；logprob 是否也应只对执行段（`ppo_agent._loss` 文档里的 `real_action_horizon` 参数其实预留了这个意图，但没实现）——这是个算法设计决策。

---

## 4. 环境/工程问题

1. **`diffusion_policy/` 子模块是空的且 `.gitmodules` 不存在**（git 里只剩 gitlink `5ba07ac`）。代码里 8+ 个文件 `from diffusion_policy.model.common.normalizer import LinearNormalizer`，新机器上克隆后必挂；`genesis_ILDP/common/normalizer.py` 是个 0 字节空文件。修复二选一：补回 `.gitmodules`（指向 columbia 的 diffusion_policy 仓库），或把用到的 `LinearNormalizer`/`CropRandomizer` 矢量化进 `genesis_ILDP/common/`。
2. `requirements.txt` 格式有问题（行尾逗号、`pip==25.0.1` 混入），`pip install -r` 会失败。
3. `agent/`、`agent/collector/`、`policy/finetune/` 等目录没有 `__init__.py`（Python3 namespace package 能容忍，但与其他有 `__init__.py` 的目录混用容易出幺蛾子，建议补齐）。

---

## 5. 今日执行清单（建议顺序）

**第一阶段：让链路能 import、能启动（机械修复，约 1-2 小时）**
- [ ] 修 3 处 `Dict[...]` 注解（§1.1）
- [ ] `critic-backup.py` 的 `Critic` 并回 `critic.py`，import 改为仓库内 CropRandomizer（§1.2）
- [ ] `ppo_agent.py`：建 `PPODiffusion`（含 `load_policy/load_critic/loss` 骨架，loss 数学先留 stub）（§1.3）
- [ ] `TrajCollector` → `TrajsCollector` 改名统一（§1.4）
- [ ] 新建 `config/finetune_diffusion_rl.yaml`（§1.5）
- [ ] FT policy `__init__` 改为透传 `**kwargs` 给父类；workspace `_target_` 路径改对（§1.6、§1.7）
- [ ] TrajCollector 五连修：eval_cfg 补 `env_runner` 节点、`task.env_runner`、`max_steps`、`episode_recording=True`、`run(self.policy)`（§1.8）
- [ ] runner `episode_buffer['action']` 第三维改 horizon，`diff_steps` 从 cfg 透传（§1.9）
- [ ] FT policy 小错四连修：`self` 断言、`'image'` key、permute、unet 转置 + `training=False` + time_encoding 批处理 + betas 设备/索引（§1.10）

**第二阶段：数据一致性（约 1 小时）**
- [ ] obs /255 统一（§2.1）
- [ ] `episode_ends` tolist、`last_obs`/`is_truncate` 接进 ReplayBuffer（§2.2、§2.3）
- [ ] 收集时关闭可视化（§2.4），核对 checkpoint 的 EMA 键（§2.5）

**第三阶段：核心算法（你来写）**
- [ ] 先定 denoising index 坐标系与微调端（§3.D）和 (t, x_t, x_{t-1}, old_logprob) 配对的数据结构（§3.E）
- [ ] 高斯 logprob（§3.A）→ PPO ratio/clip（§3.B）→ vloss + returns（§3.C）→ GAE 截断 bootstrap 与 buffer 策略（§3.F）→ logprob 作用范围（§3.G）

**搁置/标记 legacy**
- `agent/trainer/base_trainer.py`、`finetune_ppo_trainer.py`、`rl_utils.py` 的 GAE、`run_rl_finetune_dp.py`（DPPO 入口直接用 workspace 的 `main`）
