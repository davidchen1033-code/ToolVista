**长程 AgentRL 的奖励与信用分配：Qwen、Kimi、DeepSeek、MiniMax、Meta/Llama**

核查日期：2026-09-09。延续 [上一份调研](/F:/code/ToolVista/docs/agentrl_credit_assignment_research_2026-09-08.md)，本次重点补查实际奖励来源、奖励时间位置、critic 与优势计算，以及公开代码的范围。只读取论文、官方说明和作者源码；未下载模型、数据集、训练环境或安装依赖，也未运行这些工作的训练。

**主要结论：公开方法并不统一。** DeepSeek-V3.2、Kimi 的公开主体目标采用轨迹结果与组内比较；Qwen3-Coder-Next 增加明确的局部工具错误惩罚；MiniMax-M2.5 明确声称采用过程奖励，但所查材料未给出可复刻公式；Meta RLEF 则明确使用按轮次估值的 critic。不能把它们全部概括成“只评最终答案”，也不能把过程奖励、critic、token 损失、异步调度当成同一种机制。

| 工作与身份 | 奖励由什么产生 | 公开的信用分配机制 | 证据边界 |
|---|---|---|---|
| Qwen3-Coder-Next，官方 | 最终任务完成评分；未完成惩罚；无效工具调用惩罚 | 明确对无效工具调用关联 token 施加局部惩罚 | 未给出所有有效动作的独立 Q 值或完整局部项公式 [1] |
| Kimi K3 AET，官方 | 独立 verifier 评最终环境状态；预算与提交约束 | 通用 RL 沿用 K2.5；公开任务优势基于组内结果差 | 中间验证诊断不是已公开的逐步 reward；AET 具体权重、逐动作优势未披露 [2][3] |
| DeepSeek-V3.2，官方 | 推理/Agent 主要使用规则结果奖励、长度惩罚、语言一致性；部分搜索 helpful 数据使用 rubric GenRM | GRPO，任务优势为每条输出的结果分减组均值 | 不随该输出的 token 位置变化；稳定化策略不能等同于动作价值学习 [4] |
| MiniMax-M1，官方 | 可验证任务用最终正确性加格式分；软件工程用测试执行；一般任务用 GenRM | CISPO 采用 GRPO 的组相对优势 | CISPO 的 token 权重裁剪本身不产生过程分 [5] |
| MiniMax-M2.5，官方 | 明确宣布过程奖励，以及对任务完成时间的评价 | 声称过程机制用于长上下文 Agent 的信用分配，继续用 CISPO | 所查 README 未公开过程评分器输入、标签、公式、时间归属；仓库无该 trainer [6] |
| Meta RLEF，官方，Llama 3/3.1 | 最终公共与私有测试通过 +1，否则 −1；中间无效代码 −0.2；另加 KL 项 | PPO；整轮的 value critic；一轮内 token 共享该轮优势 | 明确存在跨轮估值；并非每个 token 单独评任务价值；所查材料未提供完整训练实现 [7] |
| Meta SWE-RL，官方，Llama 3 | 预测补丁与参考补丁的字符串相似度；格式错误 −1 | GRPO 组相对优势 | 连续结果分，不是执行每步后的进度分；公开 reward 实现 [8][9] |
| Meta LlamaRL，官方框架 | 实验用 SymPy 检查数学答案等规则分 | 示例采用整条生成的组均值 baseline；AIPO 处理异步策略偏差 | 框架允许 critic，不代表所有实验启用 critic；不是已公开的完整长程 Agent 配方 [10] |

**1. DeepSeek：应当分别看 V3.2、R1-Zero 与 DeepSeekMath。**

V3.2 §3 的原文是：

> “For reasoning and agent tasks, we employ rule-based outcome reward, length penalty, and language consistency reward.”

§3.1 明确给出：

\[
\hat A_{i,t}=R_i-\operatorname{mean}(R_1,\ldots,R_K).
\]

这里的 `t` 是生成 token 位置。任务优势不随 `t` 变化；该式也没有除以组内标准差，不宜照搬其他 GRPO 变体的公式。公式中的概率比同样常记作 `r_{i,t}(θ)`，那是 importance ratio，不能误读为环境即时奖励 `r_t`。[4]

V3.2 §3.2.3 使用真实搜索 API、代码工程工具、Jupyter，以及合成环境。代码环境的构建检查包括测试从失败转成功和无回归；通用合成任务设计可验证条件。搜索数据另包含 helpful 数据，其响应按多维 rubric 由 GenRM 评分，因此不能说该模型所有 Agent 样本都完全不用奖励模型。公开报告没有给出各类任务完整 reward-manager 源码及所有系数。

其 Keep Routing、Keep Sampling Mask、Off-Policy Sequence Masking 分别处理专家路由、采样动作空间、旧策略负样本的训练偏差；这些措施不提供“某次工具调用贡献了多少”的独立标签。

R1-Zero §2.2.2 明确不用神经网络 outcome/process reward model，主要采用答案正确性与格式规则。§4.2 讨论 PRM 的困难：细步骤难定义、中间正确性难验证、reward hacking 与再训练成本。这是该报告的经验，不能推论所有 DeepSeek 系列都不用 PRM。[11]

DeepSeekMath §4.1.3 则确实研究了过程监督：PRM 为每个推理步骤打分，分数归一化后，将某 token 之后各步的归一化奖励相加作为其优势：

\[
\hat A_{i,t}=\sum_{j:\operatorname{end}(i,j)\geq t}\widetilde r_{i,j}.
\]

这提供了“步骤打分→不同位置优势”的明确数学形式。它是数学推理步骤的过程监督，不能当作 V3.2 工具调用训练已经采用的算法；PRM 评价步骤与 critic 预测未来回报也应区分。[12]

**2. MiniMax：M2.5 明确有过程奖励，但公开粒度仍不够复刻。**

M1 §3.1 的 CISPO 采用组相对优势，改变重要性权重裁剪和策略损失。§4.1 明确，可验证任务采用最终正确性与格式奖励；软件工程环境运行测试，以通过/失败、编译或运行失败等结果给信号。§4.2 的 GenRM 则针对客观答案一致性和开放任务质量。[5]

M2.5 官方 README 的 “Agentic RL Algorithm and Reward Design” 写道：

> “To address the credit assignment challenge posed by long contexts in agent rollouts, we introduced a process reward mechanism for end-to-end monitoring of generation quality.”

同段还描述依据轨迹评估任务完成时间，以权衡质量与速度。这是明确的过程奖励声明，不能省略。但在所查说明中，以下细节没有给出：评分器是否为训练过的 PRM、输入是否包含完整历史、逐轮分数定义、与最终奖励如何组合、是否使用 value critic，以及优势如何计算。[6]

因此，不能擅自补成“每轮调用一个 reward LLM”“用进度差当 reward”或“使用 GAE”。仓库树只有文档、图片和许可证，没有 Forge/过程奖励 trainer；Forge 的树形样本合并描述也不能当作分支探索信用分配的证据。

补查 M2.7 官方说明与 M3 官方仓库：M2.7 提及模型辅助 RL 实验和 scaffold 优化；M3 链接的 arXiv:2606.13392 实际标题为 *MiniMax Sparse Attention*，主要讨论稀疏注意力。它们未补足此处所需的过程奖励实现，不能仅因版本较新就认为披露了新信用分配算法。[15][16]

**3. Meta RLEF：最直接回应“奖励与 critic 是否一回事”的官方例子。**

论文 *Grounding Code LLMs in Execution Feedback with Reinforcement Learning* §2.1–2.2：模型生成代码，环境运行公共测试，将错误反馈放入下一轮上下文；通过公共测试或达到轮次上限时，再用私有测试完成最终评价。[7]

论文明确列出的任务奖励为：

- 终局全部测试通过：`+1`。
- 终局任一测试失败：`−1`。
- 中间响应不包含有效代码：`−0.2`。

论文同时加入 KL 惩罚，使用 `γ=1`。终局分支优先于中间无效代码分支；不要将二者不加区分地再求和。这里给出的三项是论文实际数值，不能直接当成 ToolVista 的既定配置。

公共测试返回的报错文本供策略纠错；终局私有测试决定任务成败。这再次说明：**环境每轮都返回信息，不等于每轮都给成功进度奖励。**

RLEF 的 critic 从该轮 prompt 的最后一个 token 表示预测整轮 value；该轮输出内所有 token 使用同一个轮次优势，策略本身仍按 token 优化。附录 A.1 明确独立初始化 policy/value 网络，将 value 网络输出层替换为线性投影。§2.2 明确没有额外训练 Q 网络。[7]

这里确实有“轮次层面的价值估计”，可以避免仅用同一轨迹总分作为所有轮次相同的任务优势。但 critic 的输入是包含历史、代码、反馈的上下文，而不是单个孤立动作；任务奖励仍来自测试与规则。论文未给出足以逐行确认完整 return/GAE 实现的训练源码，不能自行指定它采用哪个 `λ` 或完整递推。

**4. Meta SWE-RL：有 reward 源码，但不是终局测试奖励。**

SWE-RL 训练模型生成代码修改，以参考补丁为目标计算序列相似度，并用 GRPO 优化。分值处于 `0–1`，格式不合法为 `−1`。这是可得到部分分数的结果奖励；连续值不等于时间上密集的逐步奖励。[8]

官方源码快照 `5aa10d67f1db07ecd3a20483d63fe2d5029ac9c9`：[9]

- `src/swerl/core/reward.py:162`，`compute_change_similarities`：遍历预测/参考涉及文件的并集。
- 第 178 行：`difflib.SequenceMatcher(..., autojunk=False).ratio()` 计算修改相似度。
- 第 227 行：对各文件相似度取平均得到结果 reward。
- 第 231 行，`calculate_search_replace_reward`：解析输出并计算奖励；第 270 行异常分支返回 `−1.0`。

这种分数无需神经网络 judge 或 critic，但依赖参考修改，且字符串相似度可能低估语义等价的其他解法；论文自身也列出这一限制。不能直接拿来评价浏览器任务是否完成。

官方仓库明确目前提供 prompt 与 reward 函数，另有 Agentless Mini 推理流程。不能将其介绍成完整开源了 Llama3-SWE-RL 的训练流水线。LlamaRL 论文则主要公开大规模异步框架设计，§6 示例是序列组均值 baseline；本次未核验到可直接运行的官方完整 LlamaRL trainer。[9][10]

**5. Qwen 与 Kimi：沿用已核对的报告，避免把不同阶段混在一起。**

Qwen3-Coder-Next §4.2.4 明确对无效工具调用对应 token 施加惩罚。这能定位局部工具错误，但尚不能证明每个语义有效动作都有独立未来价值估计。[1]

Qwen3-VL §4.5 的 Thinking with Images 采用答案准确性、多轮推理质量、工具调用评价：其中前两者分别涉及 Qwen3-32B 和 Qwen2.5-VL-72B 评审。其评价维度确实涉及过程，但报告未给出完整逐轮 reward→advantage 映射，且不能不加限定外推到所有 GUI Agent 任务。[13]

Kimi K3 AET §4.2.6 明确按最终环境状态的独立 verifier 评价给奖。公开验证接口提供诊断，隐藏验证器用于保留场景；没有公开每次 verify 都成为一个新即时 reward 的公式。[2]

K3 通用 RL 沿用 K2.5，后者公开任务优势项是组内结果差。K3 partial rollout 需要同 prompt 的采样组完成后才用于优化；不是已公布的片段末尾 critic bootstrap。K3 MOPD 阶段另有教师/学生 token 对数概率比的密集信号，那是专家蒸馏监督，不能当成 AET 验证器测得的动作贡献。[2][3]

**6. 如果目标是可读源码的动作级信用分配，ArCHer 更直接。**

ArCHer 是 UC Berkeley 等研究者的独立工作，不是 Meta 官方 Llama 配方。此版本论文主要采用 GPT-2，并有 Mistral-7B 实验，不能因为方法适用于语言模型就将其归为 Llama 训练成果。[14]

作者代码快照 `7c771a7610cb34e54ba53d69d1fb7337103e5648` 的链路如下：[17]

```python
# archer/algorithms/archer/trainer.py，摘录并省略双 Q/V 的第二支
# critic_loss: 第 66–74 行
target_v1 = reward + (1 - done) * target_v1.flatten() * self.gamma
q1_loss = self.criterion(q1, target_v1)

# update: 第 234–239 行
q1, q2, v1, v2 = self.agent.critic(batch["observation"], pi_action)
q = torch.minimum(q1, q2)
v = torch.minimum(v1, v2)
advantages = q - v
```

Q 使用实际 transition 的 `reward` 加下一状态 V 作为 TD target；V 用当前策略采样动作的目标 Q 学习。Actor 的动作优势来自 `min(Q1,Q2) − min(V1,V2)`；可选 token baseline 再降低轮内方差。这里的 Q/V 是从环境回报学习的，不是要求人事先给每个动作标注好坏。

实际 `r_t` 则看环境代码：`archer/environment/twenty_questions.py:81–85` 中，普通提问 `reward=-1`，猜中答案的终止动作 `reward=0`，还检查最大轮数。因此，它即使每步有 reward，也不是逐步运行神经网络 critic 来生成该 reward。[18]

该例也说明：失败轨迹不必所有 `r_t` 都为零；具体由任务定义决定。不能将 Twenty Questions 的计步惩罚外推为所有 ArCHer 环境的奖励。

ArCHer 公布了环境适配、critic、trainer 与配置，可作为动作级 TD 学习参考。不过这是较早、主要面向文本的小规模研究实现，迁移到 Qwen3-VL、截图与 verl 需要适配，不能称为现成的 ToolVista 插件。

**7. 作为结果奖励的源码对照：Search-R1。**

Search-R1 是第三方工作，不能因采用 Qwen 模型或名称中含 R1 就称为 Qwen/DeepSeek 官方训练实现。源码快照 `598e61bd1d36895726d28a8d06b3a15bed19f5d3`：[19]

- `verl/utils/reward_score/qa_em.py:85`，`compute_score_em` 提取最终 answer，与标准答案进行规范化精确匹配；默认正确为 1、错误或提取失败为 0。
- `verl/trainer/main_ppo.py:48` 初始化全零 reward tensor；第 78–80 行计算分数，放到最后一个有效 response token。

这是可核对的终局规则奖励路径，不会给每次检索调用单独标“检索质量”。该提取器接收 prompt+response 完整解码文本，并假设至少有两个 answer 标签匹配；迁移时需检查模板，不能直接截取一个回答就调用并假定语义相同。

**对 ToolVista 的取舍。**

已有任务首先应使用环境自己的成功条件或独立验证器，明确记录原始 `r_t`、最终任务分、无效动作原因与轮次 token 范围。以组相对结果优势建立基线，再按需要保留局部错误信号。为某任务定义 reward 的规则仍是项目职责，算法框架不能自行知道业务目标。

若重点是不同动作的长期价值，可对比 GiGPO 与轮次 critic：GiGPO 用已完成轨迹的实际后续回报，在匹配状态内做比较；ArCHer/RLEF 路线用学习到的价值函数泛化到不同上下文。GiGPO 依赖匹配覆盖率，critic 依赖学习质量与回报数据。它们均不能在没有任何有效反馈或探索的条件下凭空确定正确动作。

当前未因本次调研接入以上机制。现有 verl-agent 文件不能据此视为已完成真实环境训练验证；研究结论与工程可运行状态应分别判断。

**来源与定位。** 外链均指向本次读取的论文或官方/作者仓库。公开不足仅指本次核查范围，不声称掌握团队未公开的内部配方。

- [1] [Qwen3-Coder-Next 官方报告](https://github.com/QwenLM/Qwen3-Coder/blob/33bc6aabd7791ad7b32f7e92104f11f2359ba890/qwen3_coder_next_tech_report.pdf)，§4.2.4，第 10 页。
- [2] [Kimi K3 官方报告](https://github.com/MoonshotAI/Kimi-K3/blob/3cb39dfd32e51c3328e2e4b4af21341247d06c43/k3_tech_report.pdf)，§4.1.2–4.1.3、§4.2.6。
- [3] [Kimi K2.5，2602.02276v2](https://arxiv.org/html/2602.02276v2)，§4.4.2。
- [4] [DeepSeek-V3.2，2512.02556v1](https://arxiv.org/html/2512.02556v1)，§3、§3.1、§3.2.3。
- [5] [MiniMax-M1，2506.13585v1](https://arxiv.org/html/2506.13585v1)，§3.1、§4.1–4.2。
- [6] [MiniMax-M2.5 官方说明](https://github.com/MiniMax-AI/MiniMax-M2.5/blob/0fe00c843c16e7081a9631daeafc11288f5f871c/README.md#agentic-rl-algorithm-and-reward-design)，RL Scaling / Agentic RL Algorithm and Reward Design。
- [7] [RLEF，2410.02089v1](https://arxiv.org/html/2410.02089v1)，§2.1–2.2、附录 A.1。
- [8] [SWE-RL，2502.18449v1](https://arxiv.org/html/2502.18449v1)，§2.2–2.3、§3.6、Limitations。
- [9] [SWE-RL reward 源码](https://github.com/facebookresearch/swe-rl/blob/5aa10d67f1db07ecd3a20483d63fe2d5029ac9c9/src/swerl/core/reward.py#L162)；[公开范围说明](https://github.com/facebookresearch/swe-rl/blob/5aa10d67f1db07ecd3a20483d63fe2d5029ac9c9/README.md#-quick-start)。
- [10] [LlamaRL，2505.24034v2](https://arxiv.org/html/2505.24034v2)，§4.1、§6、§8.3。
- [11] [DeepSeek-R1，2501.12948v1](https://arxiv.org/html/2501.12948v1)，§2.2.2、§4.2；有关不用神经奖励模型的结论限定 R1-Zero。
- [12] [DeepSeekMath，2402.03300v3](https://arxiv.org/html/2402.03300v3)，§4.1.2 Outcome Supervision、§4.1.3 Process Supervision。
- [13] [Qwen3-VL，2511.21631v2](https://arxiv.org/html/2511.21631v2)，§4.5 Thinking with Images。
- [14] [ArCHer，2402.19446v1](https://arxiv.org/html/2402.19446v1)，§3.3、§5.6。
- [15] [MiniMax-M2.7 官方说明](https://github.com/MiniMax-AI/MiniMax-M2.7)。
- [16] [MiniMax-M3 官方说明](https://github.com/MiniMax-AI/MiniMax-M3)；其链接报告 [MiniMax Sparse Attention，2606.13392v1](https://arxiv.org/html/2606.13392v1)。
- [17] [ArCHer critic TD 更新](https://github.com/YifeiZhou02/ArCHer/blob/7c771a7610cb34e54ba53d69d1fb7337103e5648/archer/algorithms/archer/trainer.py#L49)；[动作优势](https://github.com/YifeiZhou02/ArCHer/blob/7c771a7610cb34e54ba53d69d1fb7337103e5648/archer/algorithms/archer/trainer.py#L234)。
- [18] [ArCHer Twenty Questions 实际奖励](https://github.com/YifeiZhou02/ArCHer/blob/7c771a7610cb34e54ba53d69d1fb7337103e5648/archer/environment/twenty_questions.py#L81)。
- [19] [Search-R1 最终答案评分](https://github.com/PeterGriffinJin/Search-R1/blob/598e61bd1d36895726d28a8d06b3a15bed19f5d3/verl/utils/reward_score/qa_em.py#L85)；[奖励放置](https://github.com/PeterGriffinJin/Search-R1/blob/598e61bd1d36895726d28a8d06b3a15bed19f5d3/verl/trainer/main_ppo.py#L78)。
