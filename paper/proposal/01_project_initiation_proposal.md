# Paper 11 课题立项开题报告替代稿

## 项目名称

基于地理空间基础模型表征的耕地适宜性认知与强化学习空间布局优化研究

## 文档用途与当前边界

本文档用于在正式论文尚未完成前，作为 Paper 11 的课题立项说明材料。它基于当前代码仓库、实验设计、真实数据适配流程和 Phase 26/27 阶段性结果整理而成，重点说明研究问题、技术路线、已有基础、阶段性结论、后续实验计划和风险控制。

当前证据尚不能支持“GeoFM 增强策略已经优于传统耕地布局优化方法”这一正式论文结论。已有实验反而表明，B1（显式规划特征 + GeoFM 64 维嵌入）相对 B0（仅显式规划特征）的 held-out learned-policy 结果仍不稳定。因此，本项目现阶段更适合定位为“GeoFM 表征是否能够为耕地空间优化提供决策相关环境信息”的机制检验与方法研发，而不是已经完成的性能宣称。

## 一句话研究论点

本研究面向耕地空间布局优化中环境适宜性信息不足的问题，构建显式规划约束与地理空间基础模型（GeoFM）潜在表征相结合的强化学习优化框架，并通过 B0/B1 主实验、随机/打乱/PCA 表征控制、适宜性代理验证和 held-out 空间评估，检验 GeoFM 嵌入是否提供超出传统 GIS 特征的决策相关信息；当前证据边界是：已有 B1-over-B0 learned-policy 结果尚未支持正向性能结论，后续需要完成表征控制与稳定性验证。

## 关键词

耕地空间优化；地理空间基础模型；AlphaEarth 嵌入；耕地适宜性代理；强化学习；MaskablePPO；空间规划决策；表征控制实验

## 术语表

| 术语 | 本项目中的含义 | 边界说明 |
|---|---|---|
| GeoFM | 地理空间基础模型，用于提供遥感学习得到的潜在环境表征 | 不等同于直接测量土壤、灌溉或肥力 |
| AlphaEarth 嵌入 | 当前使用的 64 维年度遥感嵌入特征 | 作为潜在遥感代理使用 |
| B0 | 仅使用显式规划特征的基线状态 | GIS-only baseline |
| B1 | 显式规划特征 + 原始 GeoFM 64 维嵌入 | 表征增益检验，不含适宜性奖励 |
| B2 | 显式规划特征 + 适宜性代理，使用 base + suitability reward | 当前仍未进入可用奖励阶段 |
| B3 | 显式规划特征 + GeoFM 64 维嵌入 + 适宜性代理 | 预期完整模型，尚未具备结论证据 |
| D2 | 显式规划特征 + 随机 64 维控制特征 | 控制“维度增加”而非语义信息 |
| D3 | 显式规划特征 + 打乱空间对应关系的 GeoFM 嵌入 | 控制空间对齐效应 |
| D4 | 显式规划特征 + PCA 压缩 GeoFM 嵌入 | 检验低维压缩是否足够 |
| base planning reward | 基于坡度、连片性、百亩方等规划逻辑的确定性基础奖励 | 当前已实现并用于 B0/B1 |
| suitability proxy | 弱监督适宜性代理分数 | 当前不能直接作为已验证适宜性奖励 |
| held-out tile | 与训练 tile 不同的璧山评估 tile | 仍不是跨区域迁移结论 |

## 一、研究背景与意义

耕地保护和高质量农田建设需要同时考虑坡度、连片性、地块面积、利用类型、空间邻接关系和规划可实施性。既有耕地空间布局优化模型通常依赖显式 GIS 特征和规则化规划指标，例如坡度变化、连片度提升、百亩方形成、行动合法性等。这类特征具有可解释性强、工程上可控的优点，但难以覆盖许多实际影响耕地适宜性的环境因素，例如土壤条件、灌溉便利性、长期湿润状态、作物生长稳定性、城市边缘扰动和周边生态背景。

近年来，地理空间基础模型能够从大规模多源遥感数据中学习跨区域、跨地类的潜在表征。此类表征可能携带传统规划表格中缺失的地表语义和环境背景信息，为耕地适宜性认知和空间布局优化提供新的状态表征来源。但这一设想必须经过严格实验验证：64 维嵌入是否真正带来语义信息，还是仅仅增加了模型输入维度？GeoFM 表征是否能在 held-out 空间单元上稳定改善策略行为？适宜性代理是否足够可靠，能够进入奖励函数？这些问题直接决定 Paper 11 能否从“工程实现”进入“可发表的科学结论”。

因此，本课题具有两方面意义。方法上，它探索遥感基础模型表征与强化学习空间规划的结合方式；应用上，它面向耕地保护和农田布局优化中难以获取完整土壤、灌溉和生产力数据的现实场景，尝试建立一种以显式规划约束为骨架、以遥感潜在表征为补充的可审计优化框架。

## 二、拟解决的核心问题

本项目围绕一个核心问题展开：

> 在缺少完整土壤、灌溉和生产力实测数据的情况下，GeoFM 嵌入能否作为耕地适宜性相关的潜在环境表征，提升强化学习耕地空间布局优化的状态认知和泛化能力？

该核心问题进一步分解为四个子问题：

1. GeoFM 64 维嵌入是否提供了超出显式 GIS 特征的决策相关信息？
2. B1 相对 B0 的效果是否能在多 tile、多 seed、不同训练预算下保持稳定？
3. 随机 64 维、空间打乱 GeoFM 和 PCA 压缩 GeoFM 控制组能否解释当前 B1 表现？
4. 弱监督 `suitability_proxy` 是否足以进入奖励函数，或者只能作为解释性/诊断性指标？

当前最需要解决的是第 1 和第 3 个问题。Phase 26/27 已经表明，简单延长训练预算并不能自动使 B1 优于 B0，因此下一阶段必须优先做表征控制实验，而不是直接推进 B2/B3 或跨区域迁移宣称。

## 三、研究目标

本项目拟实现以下目标：

1. 构建面向耕地空间优化的 GeoFM-enhanced block-level 状态表征，将 DLTB 地块/区块显式规划特征与 AlphaEarth 遥感嵌入进行对齐和聚合。
2. 建立可复现的 tiled held-out 强化学习评估协议，使不同 block 数量的空间单元能够在统一 padded action-mask 环境中训练与评估。
3. 比较 B0、B1 以及 D2/D3/D4 表征控制组，区分 GeoFM 语义贡献、输入维度效应、空间对齐效应和压缩表征效应。
4. 在适宜性代理进入奖励前完成弱标签验证，避免把遥感潜在表征直接解释为土壤肥力、灌溉条件或农学适宜性实测值。
5. 形成一套可审计的证据门槛：只有当 B1 稳定优于 B0 且优于关键控制组时，才进入正向论文主张；否则将论文定位为边界诊断和表征机制研究。

## 四、研究内容与技术路线

### 4.1 数据与表征构建

项目以重庆璧山区 DLTB 地类图斑及坡度数据为真实规划单元基础，将 DLTB 多边形转换为 block-to-pixel 映射，并与 AlphaEarth 年度遥感嵌入进行空间聚合。当前 Phase 11 已将 64,984 个璧山 DLTB 图斑导出为 Phase 2 兼容输入，形成显式规划特征、GeoFM block embedding 和弱标签验证所需的中间数据。

第一版 GeoFM 聚合采用 block 内像元均值：

```text
block_embedding_b = mean(z_p for p inside block b)
```

该设计避免过早引入复杂注意力或高维统计聚合，优先保证可解释、可复现和可进行控制实验。

### 4.2 强化学习优化环境

项目保留既有耕地空间规划动作语义：

```text
action = select one block for investment or adjustment
```

状态层面，B0 使用 17 维显式规划特征；B1 在 B0 基础上加入 64 维 GeoFM 嵌入；后续 B2/B3 在适宜性代理验证通过后再启用。奖励层面，当前只使用确定性 `base_planning_reward`，包括坡度、连片性、百亩方、行动合法性等规划逻辑，不启用 suitability reward。

为了适配不同 tile 的 block 数量，Phase 25 已建立 padded variable-size held-out-tile MaskablePPO 协议，通过固定最大 block 数、action mask 和全局状态补充，使训练 tile 和 held-out tile 能够在同一策略接口下运行。

### 4.3 实验分组

核心实验条件如下：

| 条件 | 状态 | 奖励 | 研究目的 |
|---|---|---|---|
| B0 | 显式规划特征 | base reward | GIS-only 基线 |
| B1 | 显式规划特征 + GeoFM 64d | base reward | 检验表征增益 |
| B2 | 显式规划特征 + suitability proxy | base + suitability reward | 检验适宜性代理奖励，当前未启用 |
| B3 | 显式规划特征 + GeoFM 64d + suitability proxy | base + suitability reward | 预期完整模型，当前未启用 |
| D2 | 显式规划特征 + 随机 64d | base reward | 控制输入维度增加 |
| D3 | 显式规划特征 + 打乱 GeoFM 64d | base reward | 控制空间语义对齐 |
| D4P8/D4P16 | 显式规划特征 + PCA 压缩 GeoFM | base reward | 检验压缩表征是否足够 |

当前 Phase 28 的设计目标正是将 D2/D3/D4 控制组接入与 B0/B1 相同的 padded held-out 协议，诊断 B1 是否具有真实表征信号。

### 4.4 评价指标

规划质量指标包括：

- total contract reward；
- 坡度改善；
- 连片性改善；
- 百亩方数量和面积变化；
- valid action rate；
- action mask violation；
- budget efficiency。

表征诊断指标包括：

- B1-B0 mean reward delta；
- B1-D2、B1-D3、B1-D4P8、B1-D4P16 mean reward delta；
- tile-seed positive count；
- positive fraction；
- sign stability；
- best comparator by mean reward。

适宜性相关指标在 reward 启用前仅作为诊断使用，包括 suitability proxy 与弱标签的分布差异、AUC/F1、坡度分位数分布和空间可视化检查。

## 五、已有工作基础

### 5.1 代码与数据工程基础

当前仓库已经从较大的 Paper58 工作区中独立出 Paper 11 reviewer package，包含设计文档、运行脚本、测试、轻量样本数据和复现实验入口。仓库可以通过 smoke check 验证基础数据读取和结构完整性。

已完成的关键工程链条包括：

- Phase 1：璧山 GeoFM 表征 baseline；
- Phase 2：block-level GeoFM feature assembly；
- Phase 8：D2/D3/D4P8/D4P16 表征控制特征表生成；
- Phase 9/10：弱标签 proxy validation 与 suitability reward readiness gate；
- Phase 11：真实璧山 DLTB 数据适配；
- Phase 12：真实 DLTB scale audit；
- Phase 13：tiled real-data contract；
- Phase 14-17：tile-level smoke env、batch smoke、baseline protocol 和 MaskablePPO readiness；
- Phase 18/19：base planning reward readiness 与基础奖励实现；
- Phase 20-25：从 same-tile pilot 到 padded held-out learned-policy pilot；
- Phase 26：B0/B1 padded held-out main empirical analysis；
- Phase 27：1024/4096 预算与 tile-seed 稳定性诊断；
- Phase 28：representation-control evaluation 已完成设计和实施计划，部分分析/writer 实现仍需根据 code review 修正后继续。

### 5.2 真实数据规模与 tiled 可行性

当前真实璧山数据链条中，Phase 11 导出了 64,984 个 DLTB 图斑。Phase 13 将其划分为 54 个非空 tile，最大 tile 包含 2,234 个 block。该规模说明 flat full-scale observation 过大，不适合作为首版 DRL 训练接口；tiled 或 padded variable-size 方案是必要选择。

### 5.3 当前主实验结果

Phase 26 对 macOS 1024-step 和 4096-step B0/B1 padded held-out 输出进行了分析。结果如下：

| 训练预算 | B1-B0 learned-policy mean delta | 支持 B1 的 tile-seed 数 | claim status |
|---|---:|---:|---|
| 1024 steps | -0.4329022862 | 4 / 9 | not_supported |
| 4096 steps | -0.1318712688 | 3 / 9 | not_supported |

Phase 27 进一步比较两个预算的稳定性，得到：

| 诊断项 | 结果 |
|---|---:|
| mean delta change | +0.3010310174 |
| positive count change | -1 |
| stable-positive | 1 |
| stable-negative | 3 |
| flip-to-positive | 2 |
| flip-to-negative | 3 |
| incomplete | 0 |
| Phase 27 status | budget_not_explanatory |

这说明增加训练预算改善了平均 delta，但未改变负向结论，且 tile-seed 符号不稳定。当前不能宣称 GeoFM 改善规划决策。

## 六、阶段性初步结论

基于目前工作进度，Paper 11 的初步结论应表述为：

1. 项目已经建立了从真实 DLTB 图斑、DEM/坡度显式特征、AlphaEarth GeoFM 嵌入到 tiled DRL 输入协议的完整工程链条。
2. 显式规划约束仍然是耕地优化任务的必要骨架，GeoFM 只能作为潜在环境表征补充，不能替代坡度、连片性和行动合法性约束。
3. 当前 B1-over-B0 learned-policy 证据为负且不稳定。4096-step 结果的 B1-B0 mean delta 为 -0.1318712688，仅 3/9 个 tile-seed pair 支持 B1。
4. 预算增加不是当前 B1 失败的充分解释。Phase 27 显示 mean delta 改善但正向 tile-seed 数下降，诊断状态为 `budget_not_explanatory`。
5. 下一步不能直接写正向论文结论，而应优先完成 D2/D3/D4 representation controls，判断 B1 的表现是否区别于随机维度、空间错配和 PCA 压缩控制。
6. `suitability_proxy` 尚未达到进入 reward 的证据门槛，B2/B3 和 suitability reward 必须暂缓。

因此，本课题目前的合理立项表述不是“已经证明 GeoFM 提升耕地优化”，而是“已经建立真实数据和可复现实验框架，并发现 B1 效果存在不稳定边界，下一阶段将通过表征控制和适宜性代理验证厘清 GeoFM 在耕地空间优化中的决策价值”。

## 七、拟创新点

1. 将地理空间基础模型嵌入引入耕地空间布局强化学习状态表征，用于弥补传统显式 GIS 特征对环境适宜性刻画不足的问题。
2. 在保持坡度、连片性、百亩方等显式规划约束的基础上，构建遥感潜在语义与规划奖励共存的保守优化框架。
3. 建立面向真实 DLTB 图斑的大规模 tiled/padded 强化学习评估协议，使不同 block 数量的空间单元可以进行 held-out learned-policy 评估。
4. 将随机 64 维、空间打乱 GeoFM 和 PCA 压缩 GeoFM 作为核心表征控制，避免把“增加输入维度”误判为“基础模型语义增益”。
5. 将 suitability proxy 置于弱标签验证和 reward-readiness gate 之后，避免对遥感嵌入做土壤、肥力、灌溉等不可支持的直接解释。

## 八、可行性分析

### 数据可行性

项目已完成真实璧山 DLTB 数据适配，并形成 Phase 2 兼容特征表。尽管原始 DLTB GeoPackage 不进入 Git 仓库，但本地工作流和复现文档已经记录其路径、派生输出和数据边界。轻量样本数据可支持 reviewer smoke test，真实数据输出可通过外部归档或 Google Drive 中转。

### 技术可行性

仓库已具备 Gymnasium/MaskablePPO action-mask 环境、padded held-out tile policy、非学习 baseline、Phase 26/27 分析器和测试体系。Phase 8 已能生成 D2/D3/D4 控制特征表，Phase 28 已完成设计和实施计划，说明表征控制评估具有明确落地路径。

### 证据可行性

现有负向结果并不削弱立项价值，反而明确了项目下一阶段的科学问题：B1 的不稳定性来自 GeoFM 表征本身、训练预算、随机种子、空间 tile 差异，还是维度/对齐控制因素？这一问题可通过 Phase 28 表征控制实验直接回答。

## 九、研究计划与进度安排

### 第一阶段：工程基础与真实数据适配（已完成）

- 完成 Paper 11 独立仓库和复现实验结构；
- 完成 AlphaEarth 样本读取、block-level feature assembly；
- 完成真实璧山 DLTB 图斑适配；
- 完成 tiled real-data contract；
- 完成基础规划奖励和 MaskablePPO readiness。

### 第二阶段：B0/B1 held-out 主实验与稳定性诊断（已完成）

- 完成 padded held-out B0/B1 policy pilot；
- 完成 Phase 26 main empirical analysis；
- 完成 Phase 27 1024/4096 budget stability diagnosis；
- 明确当前正向 B1-over-B0 claim 不成立。

### 第三阶段：表征控制实验（正在推进）

- 修正 Phase 28 analysis/writer code review 中发现的问题；
- 完成 B0/B1/D2/D3/D4P8/D4P16 同协议评估；
- 输出 B1-D2、B1-D3、B1-D4 comparator deltas；
- 判断 GeoFM raw embedding 是否区别于随机、打乱和压缩控制。

### 第四阶段：适宜性代理验证与奖励门控（计划中）

- 完成 `suitability_proxy` 与弱标签的分布、AUC/F1、坡度分位数和空间可视化检查；
- 只有当 proxy validation 通过时，才进入 B2/B3 和 suitability reward；
- 若验证不通过，则将适宜性代理降级为解释性诊断指标。

### 第五阶段：论文级证据包与正式论文撰写（计划中）

- 完成稳定 B0/B1/D 控制实验；
- 视证据情况决定是否进入 B2/B3；
- 补充空间案例图、方法图、结果表和不确定性说明；
- 将本开题报告替代稿扩展为正式论文初稿。

## 十、预期成果

1. 一套可复现的 GeoFM-enhanced farmland planning RL 实验代码和数据处理流程。
2. 一套真实 DLTB 图斑到 GeoFM block embedding 的适配与质量审计流程。
3. B0/B1/D2/D3/D4 表征控制实验结果和 reviewer-facing 证据包。
4. suitability proxy reward-readiness 诊断报告。
5. 一篇面向应用遥感、地理信息科学或空间优化方向的论文初稿。
6. 若正向证据不足，也可形成一篇强调表征边界、诊断协议和负结果价值的方法/诊断型论文。

## 十一、风险与应对

| 风险 | 当前状态 | 应对策略 |
|---|---|---|
| 将 GeoFM 过度解释为土壤/灌溉/肥力测量 | 已识别 | 统一使用“latent proxy”表述 |
| B1 不稳定或弱于 B0 | 已发生 | 进入 D2/D3/D4 表征控制，而非强行写正向结论 |
| 增加 64 维导致伪提升 | 待检验 | D2 随机 64 维控制 |
| GeoFM 空间对齐不足 | 待检验 | D3 block-shuffled 控制 |
| raw 64d 过高维且不稳定 | 待检验 | D4P8/D4P16 PCA 控制 |
| suitability proxy 不可靠 | 当前 not ready | reward gate 阻止 B2/B3 |
| 跨区域迁移证据不足 | 当前 Bishan-only | 暂不做 transfer claim，后续引入 held-out region |
| 真实数据无法直接随 Git 发布 | 已存在 | 使用轻量样本 + 外部数据归档/访问说明 |
| Phase 28 实现仍有 review 问题 | 已发现 | 修复 coverage、writer 和 status precedence 后再继续实验 |

## 十二、当前不能使用的结论

以下表述目前不能用于立项成果或正式论文结论：

- GeoFM 增强 DRL 策略已经优于传统耕地布局优化方法；
- AlphaEarth 嵌入直接测量土壤质量、灌溉条件或肥力；
- suitability reward 已经验证有效；
- B3 完整模型已经优于 B0/B1/B2；
- 当前结果已经证明跨区域迁移能力；
- Paper 11 已达到正式投稿条件。

## 十三、当前可以安全使用的立项表述

可用于立项材料的安全表述为：

> 本课题已建立真实 DLTB 图斑、显式规划特征、GeoFM 遥感嵌入与 tiled 强化学习评估协议之间的完整工程链条，并完成 B0/B1 held-out learned-policy 初步实验和预算稳定性诊断。当前结果尚不支持 GeoFM 表征相对显式规划特征的稳定正向性能结论，但已经明确了后续研究重点：通过随机、空间打乱和 PCA 压缩表征控制，判断 GeoFM 嵌入是否携带真正的决策相关环境信息；同时在适宜性代理通过弱标签验证前，不启用 suitability reward。该研究路径具有明确的问题意识、可复现实验基础和审慎的证据边界，可作为课题立项阶段的研究方案。

## 十四、结语

Paper 11 当前已经从概念设想进入真实数据和可执行实验阶段。它的阶段性价值不在于已经得到正向性能结论，而在于构建了一套能够严肃检验 GeoFM 表征是否有决策价值的实验框架，并且通过 Phase 26/27 结果暴露出 B1 不稳定这一关键科学问题。后续工作应以 Phase 28 representation-control evaluation 为核心，先回答“GeoFM 信号是否区别于维度和空间错配控制”，再决定是否推进 suitability reward、B2/B3 完整模型和跨区域迁移实验。这样的路线更适合作为课题立项依据，也能避免在正式论文阶段出现证据不足和过度宣称的问题。
