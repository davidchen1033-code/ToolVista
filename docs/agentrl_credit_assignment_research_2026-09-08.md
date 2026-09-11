**长程 AgentRL 的奖励与信用分配调研：Qwen、Kimi、GLM**

调研日期：2026-09-08。重点参考 Kimi K3 的 Autonomous Execution Tasks（AET），并核对 Qwen3-Coder-Next、Qwen3-VL、Kimi K2/K2.5、GLM-4.5/5 的官方报告，以及 GiGPO、slime 的公开代码。本文件记录研究结论和本项目的后续建议；未修改训练算法，未复现这些模型的训练。

**公开证据支持保留轨迹级结果奖励作为基线，同时把验证器、局部错误监督和训练稳定性分别处理。** 长程训练不要求先有逐动作的人工分数、过程奖励模型或 critic。部分团队确实加入局部监督，但“token 级损失”“工具反馈”“分段 rollout”都不能单独证明已经实现动作级价值估计。

| 对象 | 公开的奖励与训练机制 | 对动作信用分配能确认什么 |
|---|---|---|
| Kimi K3 AET | 独立验证器评估最终环境状态；执行中查询公开验证接口；隐藏验证器评估保留场景；提交预算与惩罚 | AET 章节没有披露专用逐动作优势公式；通用 RL 沿用 K2.5 的组比较算法 |
| Kimi K3 多教师蒸馏 | 教师与学生的逐 token 对数概率比形成密集 OPD 奖励 | 存在 token 级监督，但来源是教师策略，属于能力蒸馏阶段 |
| Qwen3-Coder-Next 软件工程专家 | 最终任务完成奖励、未完成轨迹惩罚、错误工具调用对应 token 的局部惩罚 | 明确公开局部错误监督；未公开任意有效动作的独立价值估计公式 |
| Qwen3-VL 视觉 Agent | 答案准确性奖励、多轮推理奖励、工具调用奖励 | 明确区分评价维度；未披露这些分数到每步优势的完整映射 |
| Kimi K2.5 单 Agent | 规则结果奖励、预算控制、部分任务的 GRM；组内奖励差配合 token 级裁剪和正则 | 公开目标中的任务奖励差由同一条轨迹的生成 token 共享 |
| Kimi K2.5 PARL | 冻结子 Agent，只训练调度者；任务结果加实例化奖励、子任务完成率 | 显式分离调度与执行角色，避开端到端共同优化的跨 Agent 归因歧义 |
| GLM-4.5 / GLM-5 | 轨迹组比较、模型输出 mask；GLM-5 强化异步 RL 的概率对齐与样本过滤 | 未见这些报告披露通用逐动作 critic/GAE 方案；token 裁剪处理策略偏差 |
| GiGPO / verl-agent | 轨迹优势与相同状态下的动作相对优势相加 | 有明确逐步优势公式和公开实现，不依赖额外 critic |

**Kimi K3 的 AET 是环境与验证范式。** [S1] §4.2.6（第 16 页）把每个任务定义为：初始状态、带约束的目标、工具动作空间、执行预算、独立验证器。执行时给 Agent 目标、上下文、约束与验证接口，不给该任务的参考轨迹或预定步骤。模型自行分解任务、选择工具、规划、纠错、判断何时结束。这不意味着 K3 跳过 SFT；§4.1.1 明确使用包含复杂 Agent 轨迹的 SFT 冷启动。

报告对奖励来源的原文是：

> “Rewards are grounded in the verifier’s evaluation of the final environment state rather than the agent’s self-reported completion.”

这说明奖励锚定的是环境最终状态。它不必是简单的二元成功分，也可能是验证器给出的完成程度；报告没有给出全部 AET 类型的数值公式、权重和提交惩罚系数，不能替其补全。

执行过程则允许反复提交候选方案、接收验证反馈、修改策略。典型循环是“提出假设 → 执行工具 → 提交验证 → 分析反馈 → 修正 → 最终评估”。**中途反馈能影响后续决策，但报告没有说明每次反馈都被作为独立的即时奖励送入优势估计。** 不能把 verify-in-the-loop 直接译成“逐步奖励 RL”。

公开验证器提供诊断，隐藏验证器评估未公开的保留场景；Agent 与验证器隔离，并受提交预算和惩罚约束。这使模型能够通过诊断进行探索，同时减少针对公开测试过拟合或直接干预评分逻辑的机会。[S1] §4.2.6

报告列举的 AET 类型包括黑盒系统复刻、量化因子发现、税务审计。Figure 10（第 17 页）给出 Camera Repair Management System：Agent 通过 oracle 查询，把隐藏的 3D 相机维修系统复刻成 Web 应用，图中绘制验证器评估的完成进度。**进度曲线本身不能证明训练使用了进度差作为逐步 reward。**

**K3 的三个层次需要分开看。**

| 层次 | 报告中的具体机制 | 与信用分配的关系 |
|---|---|---|
| AET 的环境反馈 | 公开验证诊断、隐藏验证、最终环境评分 | 规定任务如何交互和评分；没有完整披露逐动作优势 |
| 长程 RL 调度与更新 | partial rollout、持久化沙箱、完整组到齐后训练、token 级正则 | 支持超长轨迹和跨迭代的旧策略数据；不等于给未完成片段设置中途奖励 |
| 多教师 on-policy 蒸馏 | 按领域及推理预算选择教师，对学生采样 token 计算密集 OPD 奖励 | 提供不同 token 的教师监督；不是验证器测得的动作边际贡献 |

K3 §4.1.2（第 13 页）描述：对 N 个 prompt 各采样 K 条轨迹；当一部分轨迹完成后，暂停剩余 rollout，先进行训练，再于下一轮优先恢复。**同一个 prompt 的 K 条结果全部完成后，才将这一组派发给策略优化**。策略算法沿用 K2.5；同一条长轨迹可能跨多轮迭代，通过 per-token regularization 控制旧策略数据带来的不稳定。[S1][S2]

这与“切成片段后，用 value network 给每个片段末端 bootstrap 一个价值”是不同的机制。报告没有把 partial rollout 描述为这种 critic 方法。§5.3.2 公开了支持暂停、恢复、fork 的沙箱基础设施；支持 fork 也不能作为其使用树搜索信用分配的证据。

K2.5 §4.4.2 的公开目标含有每条响应的组内奖励差：

\[
A_j^{\mathrm{outcome}}=r(x,y_j)-\frac1K\sum_{k=1}^{K}r(x,y_k).
\]

这一项不随该轨迹中的 token 位置变化；token 概率比、梯度裁剪、正则项可以随位置变化。K3 沿用该算法的描述，支持“组级结果信号加 token 级稳定化”的判断；公开材料不足以断言它完全没有任何内部局部监督，也不足以声称它已公开 AET 专用逐动作价值学习。

K3 对 Agent 任务的预算按累计输出 token 计量，包括推理文本与工具调用参数；超出按任务设定的阈值时，把任务奖励覆盖为 −1。通过逐步收紧预算训练 max/high/low effort 专家。[S1] §4.1.2。预算惩罚优化效率，并不独立评价每一个工具动作。

K3 §4.1.3、式 (15)（第 13–14 页）公开的密集蒸馏奖励可写为：

\[
r^{d}_{\mathrm{OPD}}(y_t)=\operatorname{clip}\!\left(
\operatorname{stopgrad}\!\left[\log\frac{\pi_{\mathrm{teacher}}^{(d,e)}(y_t\mid x,y_{<t})}{\pi_\theta(y_t\mid e,x,y_{<t})}\right],-R_{\max},R_{\max}\right).
\]

教师在学生实际访问的前缀上评价该 token，再把多个领域、多个 effort 专家的能力合入统一模型。该信号能够因 token 而不同，但依赖教师质量，不能解释成“环境已经测量了每个工具动作的真实贡献”。本项目当前没有这些领域专家，也没有接入此蒸馏目标。

另一个容易混淆之处是 K3 §4.2.5 的 Personal Assistant Tasks：每个事件有自己的评估标准，使用规则或 LLM evaluator；一个 rollout 可跨模拟多日、含上千次调用。该节与 AET 分开描述，报告没有完整说明事件评分如何沿时间轴放置及其优势映射。因此不能将这些事件标准直接当成 AET 的已公开 dense reward 实现。

**Qwen 对本项目最直接的参考是局部工具错误监督。** [S3] Qwen3-Coder-Next §4.2.4（第 10 页）明确说明，软件工程专家主要按最终完成情况给轨迹奖励，同时加入两类惩罚：超过最大交互轮数的未完成轨迹惩罚，以及逐轮工具格式检查。原文写道：

> “During optimization, tokens associated with invalid tool calls receive token-level penalties.”

这比 ToolVista 当前把所有格式错误汇总到整条轨迹总分更局部：即使最后完成任务，错误调用关联的 token 也有针对性的负向监督。不过报告未披露这些局部项如何与优势或损失精确合成，不能声称它给每个语义上有效的动作都估计了独立 Q 值。报告同时强调有效任务筛选和防止从 Git 历史获取答案的验证环境治理；Figure 8 描述其训练过程中平均交互轮数由约 50 增至 130。[S3]

Qwen3-VL §4.5（Thinking with Images）则明确公开三种奖励：[S4]

| 奖励 | 评价方法 |
|---|---|
| Answer Accuracy | Qwen3-32B 判断最终答案正确性 |
| Multi-Turn Reasoning | Qwen2.5-VL-72B 判断是否正确理解工具/环境反馈，推理是否连贯 |
| Tool-Calling | 实际调用次数与离线专家根据任务复杂度估计的目标比较 |

工具调用奖励用于缓解模型退化为无论任务难度都只调用一次工具的现象。它体现任务相关的过程评价，**不代表奖励调用越多越好**。公开段落没有给出三个分量的详细系数、逐轮归属、单步优势构造。这一节主要讨论视觉工具推理，也不能无条件外推为所有 GUI 长程任务的训练配方。

Qwen 的 GSPO 论文进一步说明了另一个维度：通过序列概率比、长度归一化和序列裁剪稳定 RL，优势仍来自组内序列总奖励。[S5] §4.1，式 (5)–(7)。因此，替换 GRPO 的策略损失为 GSPO，本身不会产生逐动作价值；也不能仅凭 GSPO 属于 Qwen 就认定 Coder-Next 全部 Agent RL 使用了相同配方。

**Kimi K2.5 的角色分离适合参考未来的多 Agent 训练。** PARL 冻结子 Agent，只更新调度者；子 Agent 输出作为环境观察，不参与优化。其总奖励由最终表现、子 Agent 实例化奖励、子任务完成率组成，后两项权重在训练中退火到零。[S2] §3。这缓解了“究竟是分解错误还是执行错误”的共同优化问题，但没有声称精确还原每个子 Agent 的反事实贡献。K3 的 AET 也不能自动等同于 PARL。

Kimi K2 的 self-critic 则主要指根据 rubric 对回答进行比较和评价的 judge；仅出现 critic 一词，并不意味着存在 PPO 中估计状态价值 V(s) 的网络。[S6] §3.2.2–3.2.3

**GLM 的公开重点是保持长轨迹的训练信号正确和稳定。** GLM-4.5 §3.3 描述 group-wise 策略优化，只对模型生成 token 计算损失，排除工具与环境反馈。[S7] GLM-5 §4.1 延续轨迹组比较，并公开以下工程机制：[S8]

- 保留生成时的原始 token IDs，避免文本重新分词破坏动作与训练位置的对应关系（TITO）。
- 使用 rollout 实际 log-probability 处理一条轨迹跨多个策略版本的情况；对偏离区间的 token 屏蔽梯度。
- 按策略版本过滤过旧轨迹，排除环境崩溃造成的非模型失败，管理不完整采样组。
- 用异步调度、任务服务和持久化的轨迹记录支撑长耗时的环境交互。

这些措施可以防止将基础设施问题误归因于策略，或在错误的 token 上更新，但不是通用的“动作贡献评估器”。slime 的公开 `get_grpo_returns` 实现确实把每条样本的标量扩展到 token 维度。[S9] 框架同时支持 PPO 和自定义优势，这只能证明框架能力，不能反推 GLM 某次训练具体启用了哪一个分支。

**结果奖励本身具有统计上的信用分配能力。** 对只在末端给奖励的策略，普通策略梯度可写为：

\[
\nabla J=\mathbb{E}\left[(R-b(x))\sum_t\nabla\log\pi(a_t\mid h_t)\right].
\]

这里是解释原理的标准公式，不是上述任一模型的完整损失。虽然一条轨迹共用同一个结果信号，但动作发生在不同上下文中；在大量探索中，更常导向高回报的选择得到更多正向更新。它可以学出规划、纠错和停止策略。局限是单条轨迹会混合正确、错误和冗余动作，长轨迹与稀疏成功会使信号方差较高，有限样本下归因很粗。应区分“没有显式逐动作价值估计”与“完全没有信用分配”。

因此，上一轮讨论中的 critic + GAE 应视为可选研究路线，而不是长程 AgentRL 的必要前提；增加中间 reward 也不是唯一办法。预算、采样难度、冷启动、验证质量、环境可恢复性，都能在保留结果奖励的条件下改善学习。

**如果目标是明确实现逐动作优势，GiGPO 是更直接的开源参考。** 该研究使用 Qwen 模型开展实验，不能称为 Qwen 官方模型的训练算法。[S10] 它先按同一任务和初始状态采样多条轨迹，计算轨迹优势；再把遇到相同环境状态的动作归组，比较从这些动作出发的后续折扣回报，得到局部优势：

\[
G_{i,t}=\sum_{k=t}^{T_i}\gamma^{k-t}r_{i,k},\qquad
A_{i,t}=A_i^{\mathrm{episode}}+\lambda A_{i,t}^{\mathrm{step}}.
\]

公开代码 `gigpo/core_gigpo.py` 的 `compute_step_discounted_returns` 实现回报递推，`compute_gigpo_outcome_advantage` 明确合并 episode 和 step 两种优势。[S11] 不需要额外 value 网络，也不要求每个状态新增分支 rollout；局部信号来自已采样轨迹中可比较的状态。

这有实质限制：状态重复不足时局部信号很弱；截图相同不保证隐藏状态相同；Agent 携带不同历史时，同一画面也未必对应相同策略状态。该版代码的近似匹配仅支持文本观察。迁移到截图 Agent 时，需要定义可靠、任务相关的状态签名，并测量有效分组覆盖率，不能直接用 PNG 哈希宣称问题解决；也不能承诺已有论文在其他环境中的提升能原样复现到 MiniWoB。

Agent Lightning 同样值得借鉴 transition 数据接口，但其论文 v1 §3.3.2 明确写出，当前实现让每个动作共享最终回报，复杂信用分配留给后续扩展。[S12] 因此，仅换框架也不会自动获得更精细的动作价值。

**对 ToolVista 的建议顺序如下，均为调研后的提议。**

1. 保留现有 12 个 MiniWoB 任务作为短任务基线，先取得真实 rollout 和训练证据。它们不足以验证 K3 所强调的长程规划、反复验证和纠错能力。
2. 参考 AET 扩展任务合同：明确初始状态、目标、约束、工具 schema、执行与提交预算、公开诊断接口、独立最终验证器、验证器版本。任务行仍可由 task_id + seed/config 构造，不必首先收集每步标准答案。K3 官方库没有提供可直接替换 MiniWoB 的完整 AET 数据集与环境，具体任务规模、资源体积仍需本项目自己定义。
3. 建立结果评分与公开诊断的明确边界。隐藏验证器评估保留条件或场景；避免将 actor 自报完成、接口未报错、截图变化直接视为任务进展。若新增主动 verify 调用，还需扩展当前 JSON 动作协议和环境适配层。
4. 先参考 Qwen 的局部错误监督：保存每轮生成 token 的区间、格式/执行错误原因，以及独立的终局分量、预算分量和局部惩罚。要让错误动作受到局部更新，需要在损失或自定义优势路径保留局部项；继续求和后送入 outcome GRPO 会抹平这种定位。数值权重需要本项目实验确定。
5. 借鉴 GLM 的失败分类，把可归因于策略的无效动作与沙箱/浏览器/服务故障分开。当前 `run_episode` 的运行异常可能落成失败奖励，正式训练前应明确哪些样本应重试、过滤或整组剔除。
6. 若后续任务存在可靠重复状态，再做 GiGPO 与轨迹 GRPO 的受控对照；如果重复率低，则评估基于历史状态的 critic、可恢复环境下的分支回报估计等路线。按相同环境交互量比较成功率、预算内完成率、有效分组率与运行成本。
7. 只有在已经获得强领域教师、确有能力合并需求时，再考虑 K3 式 MOPD。它会增加教师前向与训练复杂度，不是把当前 SFT checkpoint 接入环境 RL 的必经步骤。

对应的现有代码是 [rewards.py](/F:/code/ToolVista/toolvista/agentrl/rewards.py)、[rollout.py](/F:/code/ToolVista/toolvista/agentrl/rollout.py)、[verl_agent_loop.py](/F:/code/ToolVista/toolvista/agentrl/verl_agent_loop.py) 和 [miniwob_grpo.yaml](/F:/code/ToolVista/configs/verl/miniwob_grpo.yaml)。当前仍是合并 episode reward、轨迹 GRPO、生成/观察 mask；局部惩罚对齐、AET 验证接口、GiGPO、critic、MOPD 均未因此次调研接入。

**资料与核验范围。** 引用以官方报告和作者代码为主；论文版本及代码快照如下。开源权重、公开方法描述和完整训练可复现性分别判断。K3 官方仓库在所查快照下只有 LICENSE、README、logo 和技术报告，没有 AET 任务包、验证器或 RL trainer。未获得全部私有训练配方，不将未披露内容推断为不存在。本地来源缓存位于被 Git 忽略的 `logs/agentrl-credit-research/`。

- [S1] [Kimi K3: Open Frontier Intelligence，官方 PDF](https://github.com/MoonshotAI/Kimi-K3/blob/3cb39dfd32e51c3328e2e4b4af21341247d06c43/k3_tech_report.pdf)：重点 §4.1.2–4.1.3，第 13–14 页；§4.2.5–4.2.6，第 16 页；Figure 10，第 17 页；§5.3，第 21–22 页。
- [S2] [Kimi K2.5: Visual Agentic Intelligence，2602.02276v2](https://arxiv.org/html/2602.02276v2)：2026-08-07 版本；§3 PARL、§4.4.2 Policy Optimization / Reward Function / Toggle。
- [S3] [Qwen3-Coder-Next Technical Report，官方 PDF](https://github.com/QwenLM/Qwen3-Coder/blob/33bc6aabd7791ad7b32f7e92104f11f2359ba890/qwen3_coder_next_tech_report.pdf)：报告首页日期 2026-02-03；§4.2.4，第 9–11 页。
- [S4] [Qwen3-VL Technical Report，2511.21631v2](https://arxiv.org/html/2511.21631v2)：§4.5 Thinking with Images。
- [S5] [Group Sequence Policy Optimization，2507.18071v2](https://arxiv.org/html/2507.18071v2)：§4.1–4.3。
- [S6] [Kimi K2: Open Agentic Intelligence，2507.20534v2](https://arxiv.org/html/2507.20534v2)：§3.2.2–3.2.3；critic 的含义与组级目标。
- [S7] [GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models，2508.06471v1](https://arxiv.org/html/2508.06471v1)：§3.3 Agentic RL。
- [S8] [GLM-5: from Vibe Coding to Agentic Engineering，2602.15763v2](https://arxiv.org/html/2602.15763v2)：§4.1，尤其 §4.1.2。未将后续产品能力公告等同于新的训练算法披露。
- [S9] [slime：get_grpo_returns 源码](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/utils/ppo_utils.py#L361)：代码快照 `4c193f1f37509cca70f0e88807a9305b70f63f4e`。
- [S10] [Group-in-Group Policy Optimization for LLM Agent Training，2505.10978v3](https://arxiv.org/html/2505.10978v3)：§4，§6 的状态匹配限制。
- [S11] [verl-agent：GiGPO 核心实现](https://github.com/langfengQ/verl-agent/blob/20bd331bdbc9026a5668e11362178e10ab7400c8/gigpo/core_gigpo.py#L138)：代码快照 `20bd331bdbc9026a5668e11362178e10ab7400c8`；第 87、138、334 行起的回报与优势函数。
- [S12] [Agent Lightning: Train ANY AI Agents with Reinforcement Learning，2508.03680v1](https://arxiv.org/html/2508.03680v1)：§3.3.2；只限定此论文版本的已描述实现。
