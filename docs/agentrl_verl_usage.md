# ToolVista：verl + BrowserGym MiniWoB AgentRL

本路径从现有 ToolVista LoRA checkpoint 出发，用截图执行多轮 GUI 动作，再用 MiniWoB 的任务判定训练终局 GRPO 或 GiGPO。两条路线共用 verl 0.7.0、浏览器、数据和优化配置；不需要安装 verl-agent fork。原有 `train_sft.sh`、`train_rl.sh` 和 Visual-RFT 奖励插件继续独立使用。

## 是否必须再做 SFT

**不必无条件重做一轮 SFT。** 判断依据是当前 checkpoint 是否已经学会“截图 → 可执行 GUI 动作”，以及能否偶尔完成任务。原有工具调用 JSON、静态视觉问答或 Visual-RFT GRPO 的分数不能代替这个检查。

新入口的固定 gate 是 3 个任务（click-test、click-button、enter-text）各 10 个 seed：

- JSON 解析率至少 95%；
- 已解析动作的 schema/坐标范围合法率至少 95%；
- 30 回合至少有 1 个环境判定成功；
- 没有浏览器、模型加载等基础设施错误。

通过则允许直接 AgentRL。格式不达标时，做短程 GUI SFT；格式达标但零成功时，也会阻止训练，可补 GUI 示范或先用更简单的训练课程。基础设施失败应先修环境，不能据此判定需要 SFT。这个 gate 只是冷启动门槛，不代表策略已经可靠。

## 1. AutoDL 环境

建议 Linux、Python 3.11、支持 PyTorch 2.9.0/vLLM 0.11.2 的 CUDA 驱动。依赖固定在 `configs/verl/requirements-agentrl.txt`。新 GPU 栈装进独立 venv，Swift 合并仍使用原训练 venv。

```bash
cd /root/autodl-tmp/ToolVista
# 此脚本会安装依赖与 Chromium；仅在需要配置服务器时执行。
bash scripts/autodl/setup_verl_agent.sh
export AGENTRL_PYTHON=/root/autodl-tmp/venvs/toolvista-agentrl/bin/python
```

MiniWoB 的 HTML 资源需要单独准备。固定资源 revision，避免不同机器任务实现不同：

```bash
git clone https://github.com/Farama-Foundation/miniwob-plusplus.git /root/autodl-tmp/miniwob-plusplus
git -C /root/autodl-tmp/miniwob-plusplus checkout 33c3b4ddef8c6eb67c57a29663d844b1eda7e614
export MINIWOB_URL=file:///root/autodl-tmp/miniwob-plusplus/miniwob/html/miniwob/
export MINIWOB_ASSET_REVISION=33c3b4ddef8c6eb67c57a29663d844b1eda7e614
"$AGENTRL_PYTHON" scripts/agentrl_smoke.py --preflight
```

仅接受本机 `file://` 或 loopback HTTP 资源。每个回合都有独立浏览器进程，超时后回收。MiniWoB 保持默认 500×320 viewport 和很长的墙钟期限，训练以 15 次动作上限决定超时。动态动画任务的像素未必逐次完全一致；固定 gate 使用上述简单任务。

## 2. 导出当前 checkpoint

将 `ADAPTER` 指向实际已经完成的 checkpoint。下面的路径是用法示例，应替换为服务器上的真实目录。

```bash
BASE=/root/autodl-tmp/model/Qwen3-VL-4B-Instruct
ADAPTER=/root/autodl-tmp/ToolVista/outputs/current-run/checkpoint-final
MERGED=/root/autodl-tmp/ToolVista/outputs/agentrl-init/merged-policy
SWIFT=/root/autodl-tmp/venvs/toolvista/bin/swift
PYTHON=/root/autodl-tmp/venvs/toolvista/bin/python

"$PYTHON" scripts/prepare_verl_checkpoint.py \
  --base-model "$BASE" --adapter "$ADAPTER" --output "$MERGED" \
  --swift-bin "$SWIFT" --dry-run

"$PYTHON" scripts/prepare_verl_checkpoint.py \
  --base-model "$BASE" --adapter "$ADAPTER" --output "$MERGED" \
  --swift-bin "$SWIFT"

"$AGENTRL_PYTHON" scripts/prepare_verl_checkpoint.py --output "$MERGED" --verify
```

有 retention gate 时加 `--retention-gate /absolute/path/gate.json`，它必须 `passed=true` 且授权当前 adapter。有源训练 manifest 时加 `--provenance /absolute/path/manifest.json`，要求 `authorized_checkpoint` 指向当前 adapter，或 `output_dir` 包含该 checkpoint。缺少这些可选证据不会被伪装成“已通过 retention”；合并 manifest 只记录实际提供的证据。

若 adapter 中存在 `trainer_state.json`，要求 `global_step >= max_steps > 0`。尚未训练结束的中间 checkpoint 会被拒绝。合并通过已安装的 `swift export` 执行，显式启用 safe serialization，先写暂存目录，校验完整模型后才发布。现有输出默认拒绝覆盖；确实要替换时显式加 `--overwrite`。源模型、adapter 和证据文件必须保留在原位置，以供启动时重新校验哈希；不要把源文件修改后再启动已有合并模型。

## 3. 准备任务数据并验证

```bash
DATA=/root/autodl-tmp/ToolVista/data/exports/agentrl/miniwob-v1
# 可在没有 pyarrow 的机器上确认任务清单；不写文件。
"$AGENTRL_PYTHON" scripts/prepare_agentrl_data.py --output-dir "$DATA" --dry-run
"$AGENTRL_PYTHON" scripts/prepare_agentrl_data.py --output-dir "$DATA"

bash scripts/autodl/train_agent_rl.sh --mode dry-run \
  --checkpoint "$MERGED" --data-dir "$DATA" \
  --output /root/autodl-tmp/ToolVista/outputs/agentrl/preview-01

# Transformers 本地加载精确合并模型，验证一回合截图/生成/动作/奖励。
bash scripts/autodl/train_agent_rl.sh --mode smoke \
  --checkpoint "$MERGED" --data-dir "$DATA" \
  --output /root/autodl-tmp/ToolVista/outputs/agentrl/smoke-01

# verl AgentLoop + vLLM 自身的完整 rollout，1 个 episode、每个 prompt 1 条生成轨迹；不更新权重。
# 一条轨迹最多包含 15 次动作生成。
bash scripts/autodl/train_agent_rl.sh --mode rollout \
  --checkpoint "$MERGED" --data-dir "$DATA" \
  --output /root/autodl-tmp/ToolVista/outputs/agentrl/verl-rollout-01

# 单独测是否需要 GUI SFT。
bash scripts/autodl/train_agent_rl.sh --mode gate \
  --checkpoint "$MERGED" --data-dir "$DATA" \
  --output /root/autodl-tmp/ToolVista/outputs/agentrl/gate-01
```

默认 pilot 数据是 12 个支持当前 JSON 动作协议的任务（click-test、click-button、click-link、click-option、login-user、click-tab、click-dialog、click-button-sequence、enter-text、enter-text-2、enter-password、focus-text），每个任务 100 个训练 seed，共 **1,200 行**；验证集同样 12 个任务、10 个未见 seed，共 **120 行**。先完成 100 步 pilot 和 retention 检查，再用 `--train-seeds 1000` 扩成 **12,000 行**。这只是任务/seed 配置；真正消耗时间和 GPU 的数据量是每行触发的多轮浏览器 rollout（配置默认每行采样 4 条轨迹）。seed 不同不保证画面必然不同，也不代表独立的任务类别。

固定的 MiniWoB++ revision 有 130 个 HTML 任务；第一阶段选择上述 12 个任务练习定位点击、输入、焦点和短序列操作，不能据此宣称具备通用网页 Agent 能力。未选择的任务中，一部分涉及尚未支持的动作或动态时序，一部分尚未审查奖励。特别是 `click-checkboxes`：部分正确也可能得正分，BrowserGym 0.13.3 会将其映射为成功，故先排除。

按上游该 revision 的 Git tree 文件大小统计，`miniwob/html/` 网页资源合计 4,988,688 bytes（约 4.76 MiB），全部仓库文件合计 10,753,344 bytes（约 10.26 MiB）；均不含 Git 历史、Python 依赖和 Chromium，也不是压缩包下载大小。任务配置属于 KB–MB 级小文件，Parquet 大小以实际导出为准。运行时截图、DOM 诊断和轨迹日志会另外增长。

默认 batch 为 4 行、每行 4 条轨迹。100 个训练更新约产生 **1,600 条训练 rollout**，尚未遍历完 1,200 行 pilot；完整一遍 pilot 约需 300 个更新、4,800 条训练 rollout；一遍 12,000 行约需 3,000 个更新、48,000 条训练 rollout。这些数量不包含 gate 和验证 rollout，实际动作步数取决于任务完成速度。

训练 seed 从 100 起，验证 seed 为 100000..100009，gate seed 为 0..9，三者互不重叠。数据是任务配置，不需要先生成答案；每条 row 包含 `prompt`、`agent_name=toolvista_miniwob`、`extra_info.task_id` 和 `extra_info.seed`。自定义 parquet 会检查任务标识、seed、split 重叠和 agent 名称，避免误走静态单轮 agent。

需要的转换层已经在 `prepare_agentrl_data.py`：它将任务白名单与 seed 范围展开，写出 verl `RLHFDataset` 所需字段和 Parquet。任务目标、截图和奖励在环境 reset/step 时产生，因此当前无需再套 `toolvista.adapters.BaseDatasetAdapter`。环境接口适配在 `toolvista/agentrl/miniwob_env.py`。若以后接入外部 GUI 示范数据集，再按其字段补轨迹 adapter；现有 `scripts/prepare_gui_sft.py` 只接受本项目已验证的 episode 格式，不是任意 GUI 数据集的通用转换器。

`dry-run` 仅检查 checkpoint 哈希、数据路径和新输出目录并打印命令，不检查 GPU/Chromium，也不解析 parquet 内容。实际模式再检查固定依赖、parquet 内容和本地环境。没有可用 checkpoint 的本地开发机可直接运行文末的 fake tests。

## 4. 同数据的 GRPO / GiGPO 消融

```bash
bash scripts/autodl/train_agent_rl.sh --mode train \
  --checkpoint "$MERGED" --data-dir "$DATA" \
  --output /root/autodl-tmp/ToolVista/outputs/agentrl/grpo-01 \
  --gpus 1 --steps 100

bash scripts/autodl/train_agent_gigpo.sh --mode train \
  --checkpoint "$MERGED" --data-dir "$DATA" \
  --output /root/autodl-tmp/ToolVista/outputs/agentrl/gigpo-01 \
  --gpus 1 --steps 100
```

也可以用同一个入口 `train_agent_rl.sh --algorithm gigpo ...`。先把 `--mode train` 改为 `--mode dry-run` 查看命令，再用 `--mode rollout` 验证真实浏览器与 vLLM；`rollout` 只做一条验证轨迹，不会计算组优势或更新权重。两次实验必须从相同 `$MERGED` 启动，使用相同 `$DATA` 和训练步数；完整遍历默认训练集可设 `--steps 300`。状态分组比较发生在同一更新内、同一起点的 4 条采样轨迹中，而不是跨训练更新缓存比较。

公共参数集中在 `configs/verl/miniwob_common.yaml`，两个算法配置只加载同一公共配置并选择算法。GiGPO 默认 `gamma=1`、`step_weight=1`，episode/step 都使用组内样本标准差归一化。这是实验起点，不是经过调参验证的最优值。要消融 step 权重，修改 `miniwob_gigpo.yaml` 中的 `gigpo.step_weight`；设为 0 时保留相同元数据校验，并退化为相同的 GRPO 优势。

`train` 在自己的新目录重新跑完整 gate，成功后才启动 verl。不是复用一份与模型无关的 `passed=true`。默认 4 个任务一批、每任务采样 4 条轨迹，语言模块新挂 rank-16 LoRA，视觉塔不挂 LoRA，KL 参考是合并后的起始策略。配置是小规模起步参数，未保证某一种单卡显存一定容纳；首次运行应先通过 `rollout` 模式，按实测显存调整 batch/上下文配置。默认配置总上下文上限 16384，单动作最多 256 token。

每次运行必须使用新输出目录，当前入口不做 resume。可用 `tmux` 手动托管命令。训练模型输出在 `<output>/checkpoints`；回合截图、诊断和 `episode.json` 在 `<output>/rollouts/<episode_id>`。

回合记录含 task/seed、截图路径及哈希、模型原文、解析动作、环境结果、终止原因和总奖励。DOM/可访问性树单独落盘作诊断，不送给模型。生成动作 token 的 response mask 为 1，环境观察 token 为 0。奖励直接用 `AgentLoopOutput.reward_score` 传给 verl；不调用第二个 VLM：

- 两条训练路线均为环境终局成功 1，失败/超时 0；中间所有动作 reward 为 0。
- 本次消融关闭无效动作、重复截图和循环的奖励惩罚，保留动作有效性记录与循环停止条件。
- `agentrl_smoke.py` / gate 保留历史评分口径，仅作冷启动检查，不是本次训练 reward。

`finish.status=success` 只申请结束，不会覆盖任务 evaluator 的判断。训练中的浏览器/模型调用/关闭异常会写类型化原因并报错，不会当作任务失败 0 分进入优化。

GiGPO 在 `training.json` 和 `AgentLoopOutput.extra_fields.gui_episode` 中保存每步的状态签名、原始奖励及生成 token 区间 `[start,end)`；元数据随 DataProto 重排，无需训练进程读取远程 worker 的文件。先反向计算 `G_t=r_t+gamma*G_(t+1)`，再按 `(rollout uid, state_anchor)` 分组。单元素组的 step advantage 为 0，其余按组内回报差归一化；同一动作的所有生成 token 加上该 step advantage，观察与 padding 始终 mask 为 0。

本实现采用论文的双层优势结构，但为控制变量，episode baseline 与原 GRPO 完全共用“一条轨迹一票”的实现；这与 verl-agent fork 默认按 step 行加权的 episode baseline 不同，不声称逐项复现 fork 配方。纯终局失败轨迹的 `G_t` 全为 0；若匹配组全失败，step 信号也为 0，不会凭空产生进度奖励。

状态签名来自截图哈希、目标、DOM、当前输入值/选中项、焦点和滚动位置，排除 BrowserGym 注入 ID。DOM 仅用于训练侧分组，不进入视觉策略 prompt。精确匹配偏保守；脚本隐藏状态、历史和剩余步数不保证等价，因此它仍是观察状态近似，不是因果贡献判定。控制台 `toolvista/gigpo` 日志给出 `matched_fraction`、`cross_trajectory_fraction` 和 `nonzero_fraction`；若匹配/有效信号很少，应先据此评估本路线是否有收益。

verl 0.7.0 的普通 advantage registry 不传 step 元数据，因此独立 `toolvista.agentrl.train_gigpo` 使用上游 `run_ppo(task_runner_class=...)` 入口，在自己的 Ray controller 生命周期内包裹 `compute_advantage`，退出时恢复。没有修改第三方安装目录或复制主训练循环。

## 首批任务的能力覆盖

| 任务 | 主要操作 |
|---|---|
| click-test / click-button | 定位并点击目标按钮 |
| click-link | 按文字定位链接 |
| click-option | 选择指定选项 |
| login-user | 聚焦、输入用户名/密码、提交 |
| click-tab | 切换指定标签页 |
| click-dialog | 操作弹窗按钮 |
| click-button-sequence | 按顺序点击多个按钮 |
| enter-text / enter-text-2 | 定位输入框、输入目标文本并完成任务 |
| enter-password | 密码输入控件操作 |
| focus-text | 将焦点移到指定输入框 |

这是 12 个真实 Chromium 可执行的受控网页任务，不是离线截图分类，也不是完整的真实业务网站集合。首批主要覆盖点击、焦点、输入、选项和短组合；动作协议支持滚动，但这 12 项不提供系统的长页面滚动覆盖。购物/管理后台等长工作流需后续另选可重置、有独立 verifier 的网站任务。

按当前字段将 1,320 行序列化为紧凑 JSONL 约 518 KiB；实际训练文件为 Parquet，大小由脚本导出后记录。它们只包含任务起点，不含截图/示范。每步的截图与诊断会单独占用磁盘。生成脚本不会下载资源；`MINIWOB_ASSET_REVISION` 必须与生成行中的固定版本一致，该变量是本地资源版本声明，应在确认 checkout 后设置。

## 5. gate 不通过时的短程 GUI SFT

固定 gate 的 task/seed 组合不能用于 SFT 示范，转换入口会拒绝重叠。使用不同 seed 的环境验证成功轨迹。

需要经环境验证成功的示范轨迹。可以由人工/教师策略产生，使用相同 `toolvista.episode.v1` 格式和动作协议，再通过环境 evaluator 验证；不要把失败轨迹里的动作直接当正确标签。脚本接受多个 `episode.json`：

```bash
"$PYTHON" scripts/prepare_gui_sft.py \
  --episodes /root/autodl-tmp/gui-demos/*/episode.json \
  --replay /root/autodl-tmp/ToolVista/data/exports/msswift_sft/existing_replay.jsonl \
  --replay-fraction 0.2 \
  --output /root/autodl-tmp/ToolVista/data/exports/msswift_sft/gui_bridge.jsonl

"$SWIFT" sft --model "$MERGED" --tuner_type lora \
  --dataset /root/autodl-tmp/ToolVista/data/exports/msswift_sft/gui_bridge.jsonl \
  --num_train_epochs 1 --learning_rate 1e-5 \
  --output_dir /root/autodl-tmp/ToolVista/outputs/gui-bridge-sft
```

建议 5K–20K 去重 state-action 样本，0.5–1 epoch；20% ToolVista replay 是默认起点。转换器验证 screenshot hash，保留截图历史，只监督当前目标动作，避免前序动作被重复当作多个目标。已有 replay 的消息监督设置保留。SFT 后先使用已有 retention suites 检查 router、视觉问答、工具能力，再将新 GUI LoRA（基座为本次 `$MERGED`）导出为新的合并模型，重新跑 gate。不能给新 adapter 沿用旧 checkpoint 的 retention 授权。

## 开发验证与边界

```bash
python -m unittest discover -s tests -p 'test_agentrl*.py' -v
python -m compileall -q toolvista/agentrl scripts/prepare_verl_checkpoint.py \
  scripts/agentrl_smoke.py scripts/prepare_agentrl_data.py scripts/prepare_gui_sft.py
TOOLVISTA_RUN_MINIWOB_TESTS=1 "$AGENTRL_PYTHON" -m unittest tests.test_agentrl_env_launch.RealMiniWoBSmoke -v
```

轻量测试不需要 verl/BrowserGym；截图 token 对齐测试需要 Pillow。真实 MiniWoB smoke 默认显式跳过，开启后缺依赖/浏览器就是失败。上游接口依据 verl v0.7.0、BrowserGym v0.13.3 源码，安装版本或 AgentLoop 构造/输出字段不匹配时拒绝启动。GPU、LoRA 权重同步和真实冷启动成功率只能由目标机器上的上述命令确认。
