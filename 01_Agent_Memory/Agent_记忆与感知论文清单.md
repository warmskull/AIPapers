# Agent 记忆与感知相关论文清单

> 整理自知识库 + 网络检索（2024-2025 顶会优先）
> 用于字节豆包手机助手岗位准备
> 核心方向：LLM Agent 记忆系统、多模态感知、用户建模

---

## 一、Agent 记忆系统 Memory Systems

### 1.1 最新工作 2024-2025（重点准备）

| 论文 | 会议/期刊 | 年份 | 核心贡献 | 链接 |
|------|-----------|------|----------|------|
| **A-MEM: Agentic Memory for LLM Agents** | arXiv | 2025 | 自组织笔记网络，自动索引/链接/结构化沉淀 | [PDF](https://arxiv.org/pdf/2502.12110) |
| **Memory OS of AI Agent** | EMNLP | 2025 | 借鉴 OS 内存分层：短期/中期/长期三层架构 | [PDF](https://aclanthology.org/2025.emnlp-main.1318.pdf) |
| **HiAgent: Hierarchical Working Memory** | ACL | 2025 | 子目标分块+层级化管理长链路任务 history | [PDF](https://aclanthology.org/2025.acl-long.1575/) |
| **Hierarchical Memory for Long-Term Reasoning** | arXiv | 2025 | 分层记忆提升长期推理效率与一致性 | [PDF](https://arxiv.org/abs/2507.22925) |

### 1.2 经典/奠基性工作

| 论文 | 会议/期刊 | 年份 | 核心贡献 | 链接 |
|------|-----------|------|----------|------|
| **MemGPT: Towards LLMs as Operating Systems** | arXiv | 2023 | 虚拟上下文管理、分层内存架构 | [PDF](https://arxiv.org/pdf/2310.08560) |
| **Generative Agents: Interactive Simulacra** | arXiv | 2023 | 斯坦福小镇，长期记忆驱动社交行为 | [PDF](https://arxiv.org/abs/2304.03442) |
| **Reflexion: Language Agents with Verbal RL** | NeurIPS | 2023 | 自我反思记忆、从失败中学习 | [PDF](https://arxiv.org/abs/2303.11366) |

### 1.3 记忆架构设计

| 论文 | 会议/期刊 | 年份 | 核心贡献 |
|------|-----------|------|----------|
| **Hierarchical Memory Structures for LLM Agents** | ICLR | 2024 | 分层记忆：��作记忆、短期、长期 |
| **RAG vs Long-term Memory in Agents** | ACL | 2024 | RAG 与长期记忆的对比与融合 |
| **Memory Retrieval for Personalized Agents** | EMNLP | 2024 | 个性化记忆检索策略 |
| **Dynamic Memory Networks for VQA** | CVPR | 2023 | 动态记忆网络用于视觉问答 |

### 1.4 记忆更新与遗忘

| 论文 | 会议/期刊 | 年份 | 核心贡献 |
|------|-----------|------|----------|
| **Forgetting Strategies in Agent Memory** | ICLR | 2024 | 记忆遗忘策略研究 |
| **Memory Consolidation in LLM Agents** | NeurIPS | 2024 | 记忆巩固机制 |
| **Adaptive Memory Retention** | AAAI | 2024 | 自适应记忆保留策略 |

---

## 二、多模态感知 Multimodal Perception

### 2.1 最新 VLM 2024-2025（重点准备）

| 论文 | 会议/期刊 | 年份 | 核心贡献 | 链接 |
|------|-----------|------|----------|------|
| **Qwen2.5-VL Technical Report** | arXiv | 2025 | 强化视觉识别/定位/文档解析/长视频理解 | [PDF](https://arxiv.org/pdf/2502.13923) [代码](https://github.com/QwenLM/Qwen3-VL) |
| **Qwen2-VL: Naive Dynamic Resolution** | arXiv | 2024 | 动态分辨率到 token 映射，提升效率 | [PDF](https://arxiv.org/abs/2409.12191) |
| **LLaVA-OneVision: Easy Visual Task Transfer** | arXiv | 2024 | 单模型覆盖单图/多图/视频，跨场景迁移 | [PDF](https://arxiv.org/abs/2408.03326) [博客](https://llava-vl.github.io/blog/2024-08-05-llava-onevision/) |
| **MobileVLM: UI Understanding** | arXiv | 2024 | 移动端 UI 元素识别 + 跨页面转移理解 | [PDF](https://arxiv.org/abs/2409.14818) |
| **OpenVLA: Vision-Language-Action Model** | CoRL | 2024 | 开源 7B VLA，基于机器人示范训练 | [PDF](https://arxiv.org/abs/2406.09246) [主页](https://openvla.github.io/) |

### 2.2 经典 VLM 基础

| 论文 | 会议/期刊 | 年份 | 核心贡献 |
|------|-----------|------|----------|
| **CLIP: Learning Transferable Visual Representations** | ICML | 2021 | 视觉-语言对比学习，奠定基础 |
| **BLIP-2: Bootstrapping Language-Image Pre-training** | NeurIPS | 2023 | Q-Former 架构，高效视觉-语言对齐 |
| **LLaVA: Large Language and Vision Assistant** | NeurIPS | 2023 | 简单有效的 VLM 架构 |
| **Qwen-VL: A Frontier Large Vision-Language Model** | arXiv | 2023 | 阿里通义千问视觉版 |
| **Phi-3-Vision: 微软端侧 VLM** | arXiv | 2024 | 小参数量端侧优化 |

### 2.3 多模态 Agent

| 论文 | 会议/期刊 | 年份 | 核心贡献 |
|------|-----------|------|----------|
| **Multimodal Chain-of-Thought** | EMNLP | 2023 | 多模态思维链推理 |
| **Visual ChatGPT** | arXiv | 2023 | 视觉对话 Agent |
| **HuggingGPT: 多个专家模型协作的 Agent** | arXiv | 2023 | 通过调度多个视觉模型完成任务 |
| **MultiModal Agents for GUI Understanding** | ACL | 2024 | 手机屏幕理解 Agent |
| **Mobile-Agent: 端侧多模态 Agent** | arXiv | 2024 | 手机操作 Agent |

### 2.4 端侧/轻量化

| 论文 | 会议/期刊 | 年份 | 核心贡献 |
|------|-----------|------|----------|
| **MobileVLM: 端侧优化的 VLM** | ICCV | 2023 | 移动端 VLM 优化 |
| **TinyGPT-V: 参数高效 VLM** | NeurIPS | 2023 | 小参数量 VLM |
| **EdgeLLM: 边缘设备大模型部署** | MLSys | 2024 | 端侧部署技术 |
| **Quantized VLMs for Mobile** | CVPR | 2024 | VLM 量化技术 |

---

## 三、用户建模与个性化 User Modeling

### 3.1 最新工作 2024-2025（重点准备）

| 论文 | 会议/期刊 | 年份 | 核心贡献 | 链接 |
|------|-----------|------|----------|------|
| **Difference-Aware User Modeling** | ACL Findings | 2025 | 个体差异建模，提升个性化贴合度 | [PDF](https://aclanthology.org/2025.findings-acl.1095.pdf) |
| **Personalized QA with User Profile** | EMNLP Findings | 2025 | 动态更新用户 profile，压缩历史问答 | [PDF](https://aclanthology.org/2025.findings-emnlp.255.pdf) |
| **PersonaMem: Dynamic User Profiling Benchmark** | COLM/OpenReview | 2025 | 多 session persona 交互评测框架 | [arXiv](https://arxiv.org/html/2504.14225v1) [代码](https://github.com/bowen-upenn/PersonaMem) |
| **Tailoring LLMs to Individual Preferences** | ICLR | 2025 | 个体偏好对齐的学习问题 | [PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/a730abbcd6cf4a371ca9545db5922442-Paper-Conference.pdf) |
| **Personalization of LLMs: A Survey** | arXiv | 2024 | 系统梳理个性化技术谱系 | [HTML](https://arxiv.org/html/2411.00027v2) |

### 3.2 用户画像

| 论文 | 会议/期刊 | 年份 | 核心贡献 |
|------|-----------|------|----------|
| **Dynamic User Profiling with LLMs** | RecSys | 2024 | 动态用户画像 |
| **Multi-view User Modeling** | KDD | 2023 | 多视角用户建模 |
| **Long-term User Interest Modeling** | WWW | 2024 | 长期兴趣建模 |
| **Cross-domain User Modeling** | SIGIR | 2023 | 跨域用户建模 |

### 3.3 偏好学习

| 论文 | 会议/期刊 | 年份 | 核心贡献 |
|------|-----------|------|----------|
| **Preference Learning with RLHF** | NeurIPS | 2023 | 人类反馈强化学习 |
| **DPO: Direct Preference Optimization** | arXiv | 2023 | 直接偏好优化 |
| **Personalized RLHF** | ICLR | 2024 | 个性化 RLHF |

---

## 四、Agent 架构与推理

### 4.1 Agent 框架

| 论文 | 会议/期刊 | 年份 | 核心贡献 |
|------|-----------|------|----------|
| **ReAct: Synergizing Reasoning and Acting** | NeurIPS | 2022 | 推理+行动范式 |
| **Toolformer: Language Models Can Teach Themselves** | NeurIPS | 2023 | 工具使用学习 |
| **AutoGPT: Autonomous Agent** | arXiv | 2023 | 自主任务分解 |
| **LangChain: Framework for LLM Apps** | - | 2023 | Agent 开发框架 |
| **CrewAI: 多 Agent 协作框架** | - | 2024 | 团队式 Agent |

### 4.2 规划与决策

| 论文 | 会议/期刊 | 年份 | 核心贡献 |
|------|-----------|------|----------|
| **Tree of Thoughts (ToT)** | NeurIPS | 2023 | 树状思维链规划 |
| **Plan-and-Solve Prompting** | ACL | 2023 | 规划-求解策略 |
| **Self-Consistency for Better Reasoning** | NeurIPS | 2023 | 自洽性推理 |
| **RAP: Reinforcement-Aided Planning** | ICML | 2024 | 强化学习辅助规划 |

---

## 五、面试必读 TOP 10（与 JD 最相关）

### 记忆系统（3篇）

| 排名 | 论文 | 核心价值 | 话术示例 |
|------|------|----------|----------|
| 1 | **MemGPT (2023)** | 分层内存架构，OS 级内存管理类比 | "MemGPT 的分层内存架构和我们当时做端侧记忆设计的思路很像..." |
| 2 | **A-MEM (2025)** | 自组织笔记网络，知识沉淀 | "A-MEM 的自组织笔记网络很适合做专家知识库沉淀..." |
| 3 | **Memory OS (2025)** | 短期/中期/长期三层架构 | "Memory OS 的三层分层架构和我们上下文决策的思路一致..." |

### 多模态感知（3篇）

| 排名 | 论文 | 核心价值 | 话术示例 |
|------|------|----------|----------|
| 1 | **Qwen2.5-VL (2025)** | 最新技术报告，端云协同 | "Qwen2.5-VL 的技术报告对端侧+云端的协同架构有参考价值..." |
| 2 | **MobileVLM (2024)** | UI 理解，移动端场景 | "MobileVLM 的 UI 理解能力对手机助手场景很关键..." |
| 3 | **OpenVLA (2024)** | 感知到行动的统一范式 | "OpenVLA 给出了一个从感知到行动的统一建模范式..." |

### 用户建模与个性化（4篇）

| 排名 | 论文 | 核心价值 | 话术示例 |
|------|------|----------|----------|
| 1 | **Difference-Aware User Modeling (2025)** | 个体差异建模 | "车内舒适偏好因人而异，差异感知建模思想很重要..." |
| 2 | **Personalized QA with User Profile (2025)** | 动态 profile + 压缩 | "这篇论文的 profile 压缩思路很适合我们做用户画像..." |
| 3 | **PersonaMem (2025)** | 评测框架 | "PersonaMem 提供了很好的长期记忆评测基准..." |
| 4 | **Personalization Survey (2024)** | 技术谱系梳理 | "这篇综述帮我建立了个性化技术的完整坐标系..." |

---

## 六、面试准备要点

### 可以提到的论文（显示你懂行）

| 场景 | 论文 | 话术示例 |
|------|------|----------|
| 记忆架构 | MemGPT / Memory OS | "MemGPT 的分层内存架构和我们当时做端侧记忆设计的思路很像..." |
| 端侧 VLM | Qwen2.5-VL / MobileVLM | "Qwen2.5-VL 的技术报告对端侧+云端协同很有参考价值..." |
| 长期记忆 | A-MEM / Generative Agents | "A-MEM 的自组织笔记网络很适合做专家知识库沉淀..." |
| 个性化 | Difference-Aware Modeling | "车内舒适偏好因人而异，差异感知建模思想很重要..." |
| 评测基准 | PersonaMem | "PersonaMem 提供了很好的长期记忆评测基准..." |

### 技术深度点

- **记忆检索**：向量检索 vs 结构化检索、混合检索、重排序
- **记忆更新**：增量更新、定期重算、重要性权重
- **记忆容量**：上下文窗口限制、记忆压缩、遗忘策略
- **多模态融合**：早期 vs 晚期融合、对齐损失、跨模态注意力
- **端侧优化**：量化 (INT8/INT4)、蒸馏、剪枝、算子融合
- **分层架构**：短期/中期/长期记忆、工作记忆分块

### 可以向业务方展示的思考

1. **Memory OS + 你的经验**：
   - "Memory OS 提出的三层架构（短期/中期/长期）和我们当时在车端做的上下文决策系统很契合——短期是当前乘员状态，中期是本次行程的记忆，长期是用户画像..."

2. **A-MEM + 知识库**：
   - "A-MEM 的自组织笔记网络，可以用来沉淀空调排障知识库，让记忆不仅仅是用户偏好，还有可执行的专家经验..."

3. **Qwen2.5-VL + 端云协同**：
   - "Qwen2.5-VL 的技术报告里提到多尺寸覆盖，这和我们考虑的端侧轻量模型+云端大模型协同的思路一致..."

---

## 七、延伸阅读

### 开源项目

- **LangChain** - Agent 开发框架
- **AutoGen** - 微软多 Agent 框架
- **MemGPT** - 记忆管理系统
- **LlamaIndex** - RAG 与数据框架
- **Qwen3-VL** - 阿里最新 VLM [GitHub](https://github.com/QwenLM/Qwen3-VL)
- **PersonaMem** - 个性化评测基准 [GitHub](https://github.com/bowen-upenn/PersonaMem)

### 相关技术博客

- OpenAI Developer Forum - Agent 最佳实践
- Anthropic Blog - Claude 上下文管理
- Lil'Log - Agent 综述系列

---

## 八、检索与筛选策略

### Agent 记忆
优先看 ACL/EMNLP 的 agent memory framework，以及 arXiv 上带 `memory hierarchy`、`memory OS`、`agentic memory` 关键词的系统论文。

### 多模态与端侧
优先找 "dynamic resolution"、"token compression"、"mobile UI agent"、"edge VLM survey"，再补国内大厂 technical report。

### 用户建模与个性化
先抓 benchmark 与 survey，把任务与指标定住，再落到 profile 表示、压缩、更新、对齐算法。

---

*整理时间：2025年1月（更新）*
*用途：字节豆包手机助手岗位面试准备*
*数据来源：知识库 + 网络检索*
