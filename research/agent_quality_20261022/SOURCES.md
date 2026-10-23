# Источники исследования качества агента

Дата проверки: 22 октября 2026 года. Выборка целевая: популярность плюс полезность для ML. Это не рейтинг всех агентов GitHub. Звёзды служат способом отбора и не доказывают качество. Зафиксированы 27 репозиториев; получен 41 файл, включая исходный код из 24 репозиториев. Чтение выборочное, по конкретным механизмам; проекты не запускались и не проходили полный аудит.

Числа и версии: [github_repositories.json](github_repositories.json). Фрагменты кода с исходными номерами строк: [code_evidence.json](code_evidence.json). Отдельное свидетельство Optuna: [optuna_421_evidence.json](optuna_421_evidence.json).

## Репозитории

| Репозиторий | Звёзды в снимке | Изученные файлы |
|---|---:|---|
| [WecoAI/aideml](https://github.com/WecoAI/aideml) | 1,591 | [C01](#c01), [C39](#c39), [C40](#c40) |
| [SWE-agent/mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) | 8,732 | [C02](#c02) |
| [SWE-agent/SWE-agent](https://github.com/SWE-agent/SWE-agent) | 20,612 | [C03](#c03) |
| [OpenHands/OpenHands](https://github.com/OpenHands/OpenHands) | 91,543 | Только метаданные/структура |
| [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) | 43,706 | [C06](#c06) |
| [microsoft/autogen](https://github.com/microsoft/autogen) | 61,522 | [C07](#c07) |
| [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI) | 59,979 | [C32](#c32), [C33](#c33) |
| [huggingface/smolagents](https://github.com/huggingface/smolagents) | 29,978 | [C08](#c08) |
| [Aider-AI/aider](https://github.com/Aider-AI/aider) | 49,692 | [C09](#c09), [C10](#c10) |
| [cline/cline](https://github.com/cline/cline) | 70,708 | [C11](#c11) |
| [langchain-ai/deepagents](https://github.com/langchain-ai/deepagents) | 30,503 | [C12](#c12), [C13](#c13), [C14](#c14) |
| [microsoft/RD-Agent](https://github.com/microsoft/RD-Agent) | 15,018 | [C15](#c15), [C16](#c16), [C31](#c31) |
| [SakanaAI/AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2) | 7,381 | [C17](#c17) |
| [facebookresearch/aira-dojo](https://github.com/facebookresearch/aira-dojo) | 177 | [C18](#c18), [C19](#c19), [C20](#c20) |
| [FoundationAgents/MetaGPT](https://github.com/FoundationAgents/MetaGPT) | 70,962 | [C23](#c23) |
| [Significant-Gravitas/AutoGPT](https://github.com/Significant-Gravitas/AutoGPT) | 187,699 | [C27](#c27) |
| [FoundationAgents/OpenManus](https://github.com/FoundationAgents/OpenManus) | 58,755 | [C24](#c24) |
| [SJTU-SAI-agents/ML-Master](https://github.com/SJTU-SAI-agents/ML-Master) | 455 | [C21](#c21) |
| [OpenHands/software-agent-sdk](https://github.com/OpenHands/software-agent-sdk) | 1,289 | [C04](#c04), [C05](#c05) |
| [bytedance/trae-agent](https://github.com/bytedance/trae-agent) | 12,156 | [C30](#c30) |
| [letta-ai/letta](https://github.com/letta-ai/letta) | 25,226 | [C29](#c29) |
| [stanfordnlp/dspy](https://github.com/stanfordnlp/dspy) | 38,729 | [C22](#c22) |
| [OpenAutoCoder/Agentless](https://github.com/OpenAutoCoder/Agentless) | 2,130 | [C25](#c25) |
| [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) | 256,091 | [C28](#c28), [C41](#c41) |
| [google/adk-recipes](https://github.com/google/adk-recipes) | 10,519 | [C34](#c34), [C35](#c35), [C36](#c36), [C37](#c37), [C38](#c38) |
| [aaif-goose/goose](https://github.com/aaif-goose/goose) | 55,523 | [C26](#c26) |
| [jaehyun513/MLE-STAR](https://github.com/jaehyun513/MLE-STAR) | 17 | Только метаданные/структура |

OpenHands/OpenHands в снимке в основном содержит интерфейс; исполняющий агент отдельно проверен в OpenHands/software-agent-sdk. MLE-STAR читается по историческому commit: пример удалён из основной ветки Google в июле 2026 года. Letta в этой выборке изучен только по README, без выводов о внутренней реализации. Авторский jaehyun513/MLE-STAR использован для поиска происхождения примеров, исполняющий код проверен в историческом Google sample. Переименования block/goose → aaif-goose/goose и google/adk-samples → google/adk-recipes проверены.

## Статьи и документация

### P01 — AIRA-dojo: AI Research Agents for Machine Learning

[Первичный источник](https://arxiv.org/html/2507.02554v1). Проверено: Controlled memory/operator/search ablations; proxy generalization gap; 24h H200 environment. Граница вывода: Not Qwen 27B, CPU 3h, or this contest.

### P02 — MLE-STAR

[Первичный источник](https://arxiv.org/html/2506.15692v2). Проверено: Targeted block refinement, external model retrieval, ensemble and leakage ablations. Граница вывода: Google article Lite 64% differs from later abstract full 44%; report uses versioned Table 4, no blended headline.

### P03 — MLE-bench

[Первичный источник](https://arxiv.org/html/2410.07095v3). Проверено: Scaffold comparison, pass@k vs pass@1, time/compute and contamination limits. Граница вывода: Oracle pass@k is not implementable validation selection and uses more compute.

### P04 — Executable Code Actions Elicit Better LLM Agents (CodeAct)

[Первичный источник](https://arxiv.org/abs/2402.01030). Проверено: Code as compositional action space with execution feedback. Граница вывода: General action benchmarks do not prove unrestricted shell improves this ML agent.

### P05 — GEPA

[Первичный источник](https://arxiv.org/abs/2507.19457). Проверено: Reflective prompt optimization using traces and measured task feedback. Граница вывода: Different tasks/backbones; prompt optimizer is not a contest-time ML optimizer.

### P06 — Towards a Science of Scaling Agent Systems

[Первичный источник](https://arxiv.org/abs/2512.08296). Проверено: Architecture comparison under standardized token budgets; task-dependent coordination overhead. Граница вывода: Non-ML benchmarks; no transfer of percentages to this contest.

### P07 — Large Language Models Cannot Self-Correct Reasoning Yet

[Первичный источник](https://arxiv.org/abs/2310.01798). Проверено: Limits of intrinsic self-correction without external feedback. Граница вывода: 2023 reasoning results are a caution, not proof about current Qwen.

### E01 — Effective context engineering

[Первичный источник](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents). Проверено: Just-in-time context, structured notes, compaction. Граница вывода: Engineering observations, no controlled ML effect for our agent.

### E02 — Writing effective tools for agents

[Первичный источник](https://www.anthropic.com/engineering/writing-tools-for-agents). Проверено: Tool ergonomics, meaningful bounded outputs, evaluation. Граница вывода: Do not replace deterministic tools with hidden LLM reasoning.

### E03 — Multi-agent research system

[Первичный источник](https://www.anthropic.com/engineering/multi-agent-research-system). Проверено: Parallel research and substantial token overhead. Граница вывода: Web research is not sequential CPU ML training.

### E04 — SWE-agent ACI

[Первичный источник](https://github.com/SWE-agent/SWE-agent/blob/3ea751c087f32b16e039a2233dd6eefecef325d5/docs/background/aci.md). Проверено: Lint gate, bounded file viewer, succinct search. Граница вывода: SWE patches not ML quality.

### H01 — Optuna TPESampler 4.2.1

[Первичный источник](https://optuna.readthedocs.io/en/v4.2.1/reference/samplers/generated/optuna.samplers.TPESampler.html). Проверено: n_startup_trials=10; initial independent samples are random. Граница вывода: Reducing startup does not guarantee improved ML quality.

### H02 — Optuna GridSampler 4.2.1

[Первичный источник](https://optuna.readthedocs.io/en/v4.2.1/reference/samplers/generated/optuna.samplers.GridSampler.html). Проверено: Finite Cartesian grid enumeration; grid values are not automatically quantized. Граница вывода: Grid is not automatically more efficient than TPE/random.

### H03 — scikit-learn 1.5.2 hyperparameter search

[Первичный источник](https://scikit-learn.org/1.5/modules/grid_search.html). Проверено: Grid/random search; fit and evaluation structure. Граница вывода: Use explicit temporal/group splits instead of default random CV.

### H04 — Optuna TPE source pinned v4.2.1

[Первичный источник](https://github.com/optuna/optuna/blob/v4.2.1/optuna/samplers/_tpe/sampler.py#L274). Проверено: Default startup 10 and random path at lines 408-411. Граница вывода: Newer releases need independent checks.

### H05 — FLAML economical optimization

[Первичный источник](https://microsoft.github.io/FLAML/docs/Use-Cases/Tune-User-Defined-Function/). Проверено: CFO and cost-related hyperparameters. Граница вывода: Current actual adapter selection is established by local code, not doc defaults.

## Закреплённые файлы кода

Номера строк относятся к указанной версии. Полный файл доступен по постоянной ссылке; локально сохранены только короткие фрагменты и указатели. Наличие механизма в коде подтверждает его реализацию, но не величину выигрыша.

### C01

[WecoAI/aideml — aide/utils/config.yaml](https://github.com/WecoAI/aideml/blob/ad73d227af481bea3de06f884f41b2993dd6453a/aide/utils/config.yaml)

Версия: `ad73d227af481bea3de06f884f41b2993dd6453a`; строк: 72. Git blob: `77ecfb5ecee7675acf8ad2132923514e92b6fbc2`. Проверены структура и доступные определения; дополнительные фрагменты приведены в тексте исследования.

### C02

[SWE-agent/mini-swe-agent — src/minisweagent/agents/default.py](https://github.com/SWE-agent/mini-swe-agent/blob/04d809ceab9df28f9adaed044884180159172930/src/minisweagent/agents/default.py)

Версия: `04d809ceab9df28f9adaed044884180159172930`; строк: 191. Git blob: `ca310e37d7c61888c576f73f5e1790ac769407b1`. Опорные фрагменты: [88–109](https://github.com/SWE-agent/mini-swe-agent/blob/04d809ceab9df28f9adaed044884180159172930/src/minisweagent/agents/default.py#L88-L109), [126–147](https://github.com/SWE-agent/mini-swe-agent/blob/04d809ceab9df28f9adaed044884180159172930/src/minisweagent/agents/default.py#L126-L147), [130–151](https://github.com/SWE-agent/mini-swe-agent/blob/04d809ceab9df28f9adaed044884180159172930/src/minisweagent/agents/default.py#L130-L151), [154–175](https://github.com/SWE-agent/mini-swe-agent/blob/04d809ceab9df28f9adaed044884180159172930/src/minisweagent/agents/default.py#L154-L175).

### C03

[SWE-agent/SWE-agent — sweagent/agent/agents.py](https://github.com/SWE-agent/SWE-agent/blob/3ea751c087f32b16e039a2233dd6eefecef325d5/sweagent/agent/agents.py)

Версия: `3ea751c087f32b16e039a2233dd6eefecef325d5`; строк: 1295. Git blob: `4844aeb7c42e06121fe37ba223e9263c04b53c3b`. Опорные фрагменты: [237–258](https://github.com/SWE-agent/SWE-agent/blob/3ea751c087f32b16e039a2233dd6eefecef325d5/sweagent/agent/agents.py#L237-L258), [239–260](https://github.com/SWE-agent/SWE-agent/blob/3ea751c087f32b16e039a2233dd6eefecef325d5/sweagent/agent/agents.py#L239-L260), [328–349](https://github.com/SWE-agent/SWE-agent/blob/3ea751c087f32b16e039a2233dd6eefecef325d5/sweagent/agent/agents.py#L328-L349), [390–411](https://github.com/SWE-agent/SWE-agent/blob/3ea751c087f32b16e039a2233dd6eefecef325d5/sweagent/agent/agents.py#L390-L411), [1235–1256](https://github.com/SWE-agent/SWE-agent/blob/3ea751c087f32b16e039a2233dd6eefecef325d5/sweagent/agent/agents.py#L1235-L1256).

### C04

[OpenHands/software-agent-sdk — openhands-sdk/openhands/sdk/agent/agent.py](https://github.com/OpenHands/software-agent-sdk/blob/efec2c38cda179129bef4e93dcf5f66ebe3695d4/openhands-sdk/openhands/sdk/agent/agent.py)

Версия: `efec2c38cda179129bef4e93dcf5f66ebe3695d4`; строк: 1650. Git blob: `86570898f76ced3b7b2dbbfe02713d6e101be641`. Опорные фрагменты: [690–711](https://github.com/OpenHands/software-agent-sdk/blob/efec2c38cda179129bef4e93dcf5f66ebe3695d4/openhands-sdk/openhands/sdk/agent/agent.py#L690-L711).

### C05

[OpenHands/software-agent-sdk — openhands-sdk/openhands/sdk/context/condenser/llm_summarizing_condenser.py](https://github.com/OpenHands/software-agent-sdk/blob/efec2c38cda179129bef4e93dcf5f66ebe3695d4/openhands-sdk/openhands/sdk/context/condenser/llm_summarizing_condenser.py)

Версия: `efec2c38cda179129bef4e93dcf5f66ebe3695d4`; строк: 567. Git blob: `08b586ec9c78915130aee51675992a61dd1c36df`. Опорные фрагменты: [140–161](https://github.com/OpenHands/software-agent-sdk/blob/efec2c38cda179129bef4e93dcf5f66ebe3695d4/openhands-sdk/openhands/sdk/context/condenser/llm_summarizing_condenser.py#L140-L161), [412–433](https://github.com/OpenHands/software-agent-sdk/blob/efec2c38cda179129bef4e93dcf5f66ebe3695d4/openhands-sdk/openhands/sdk/context/condenser/llm_summarizing_condenser.py#L412-L433).

### C06

[langchain-ai/langgraph — libs/langgraph/langgraph/pregel/_loop.py](https://github.com/langchain-ai/langgraph/blob/3cd57c038ad827d05f80c6488adea9994156c58c/libs/langgraph/langgraph/pregel/_loop.py)

Версия: `3cd57c038ad827d05f80c6488adea9994156c58c`; строк: 2167. Git blob: `4f3d21538f25c8e87ab4ef0c4c233ecd74f4974e`. Опорные фрагменты: [827–848](https://github.com/langchain-ai/langgraph/blob/3cd57c038ad827d05f80c6488adea9994156c58c/libs/langgraph/langgraph/pregel/_loop.py#L827-L848), [840–861](https://github.com/langchain-ai/langgraph/blob/3cd57c038ad827d05f80c6488adea9994156c58c/libs/langgraph/langgraph/pregel/_loop.py#L840-L861).

### C07

[microsoft/autogen — python/packages/autogen-agentchat/src/autogen_agentchat/teams/_group_chat/_magentic_one/_magentic_one_orchestrator.py](https://github.com/microsoft/autogen/blob/004749e6b7102c8fd317b9b1499ce5807223c278/python/packages/autogen-agentchat/src/autogen_agentchat/teams/_group_chat/_magentic_one/_magentic_one_orchestrator.py)

Версия: `004749e6b7102c8fd317b9b1499ce5807223c278`; строк: 537. Git blob: `176789257ba792c7b7b0151d002730dacd0bb7b2`. Опорные фрагменты: [300–321](https://github.com/microsoft/autogen/blob/004749e6b7102c8fd317b9b1499ce5807223c278/python/packages/autogen-agentchat/src/autogen_agentchat/teams/_group_chat/_magentic_one/_magentic_one_orchestrator.py#L300-L321), [402–423](https://github.com/microsoft/autogen/blob/004749e6b7102c8fd317b9b1499ce5807223c278/python/packages/autogen-agentchat/src/autogen_agentchat/teams/_group_chat/_magentic_one/_magentic_one_orchestrator.py#L402-L423), [451–472](https://github.com/microsoft/autogen/blob/004749e6b7102c8fd317b9b1499ce5807223c278/python/packages/autogen-agentchat/src/autogen_agentchat/teams/_group_chat/_magentic_one/_magentic_one_orchestrator.py#L451-L472).

### C08

[huggingface/smolagents — src/smolagents/agents.py](https://github.com/huggingface/smolagents/blob/96f33faaf028479119ec8d34507b47694cf14e34/src/smolagents/agents.py)

Версия: `96f33faaf028479119ec8d34507b47694cf14e34`; строк: 1814. Git blob: `a8f6e52de0361a6c23c0aa5d5907180969f259b7`. Опорные фрагменты: [436–457](https://github.com/huggingface/smolagents/blob/96f33faaf028479119ec8d34507b47694cf14e34/src/smolagents/agents.py#L436-L457), [782–803](https://github.com/huggingface/smolagents/blob/96f33faaf028479119ec8d34507b47694cf14e34/src/smolagents/agents.py#L782-L803).

### C09

[Aider-AI/aider — aider/repomap.py](https://github.com/Aider-AI/aider/blob/5dc9490bb35f9729ef2c95d00a19ccd30c26339c/aider/repomap.py)

Версия: `5dc9490bb35f9729ef2c95d00a19ccd30c26339c`; строк: 868. Git blob: `541bba6ef4a5385b8cf032201ec6e3f3e32a6ea6`. Опорные фрагменты: [525–546](https://github.com/Aider-AI/aider/blob/5dc9490bb35f9729ef2c95d00a19ccd30c26339c/aider/repomap.py#L525-L546), [529–550](https://github.com/Aider-AI/aider/blob/5dc9490bb35f9729ef2c95d00a19ccd30c26339c/aider/repomap.py#L529-L550).

### C10

[Aider-AI/aider — aider/coders/architect_coder.py](https://github.com/Aider-AI/aider/blob/5dc9490bb35f9729ef2c95d00a19ccd30c26339c/aider/coders/architect_coder.py)

Версия: `5dc9490bb35f9729ef2c95d00a19ccd30c26339c`; строк: 49. Git blob: `f3e2a38b13a4f964dd532db2de75fc9edbe636e3`. Опорные фрагменты: [6–27](https://github.com/Aider-AI/aider/blob/5dc9490bb35f9729ef2c95d00a19ccd30c26339c/aider/coders/architect_coder.py#L6-L27).

### C11

[cline/cline — sdk/packages/core/src/extensions/context/summarize-conversation.ts](https://github.com/cline/cline/blob/c441897f11f658f978a00678f045670c1ab2b60d/sdk/packages/core/src/extensions/context/summarize-conversation.ts)

Версия: `c441897f11f658f978a00678f045670c1ab2b60d`; строк: 178. Git blob: `d53e4dab689c516d64338a7c422670ac4236927c`. Опорные фрагменты: [26–47](https://github.com/cline/cline/blob/c441897f11f658f978a00678f045670c1ab2b60d/sdk/packages/core/src/extensions/context/summarize-conversation.ts#L26-L47).

### C12

[langchain-ai/deepagents — libs/deepagents/deepagents/graph.py](https://github.com/langchain-ai/deepagents/blob/b4ac8829692785a221150658401ae558f4bff3f3/libs/deepagents/deepagents/graph.py)

Версия: `b4ac8829692785a221150658401ae558f4bff3f3`; строк: 1018. Git blob: `5e08084b17ccbfc051ab891e35508f884813d4f7`. Опорные фрагменты: [724–745](https://github.com/langchain-ai/deepagents/blob/b4ac8829692785a221150658401ae558f4bff3f3/libs/deepagents/deepagents/graph.py#L724-L745), [830–851](https://github.com/langchain-ai/deepagents/blob/b4ac8829692785a221150658401ae558f4bff3f3/libs/deepagents/deepagents/graph.py#L830-L851), [895–916](https://github.com/langchain-ai/deepagents/blob/b4ac8829692785a221150658401ae558f4bff3f3/libs/deepagents/deepagents/graph.py#L895-L916), [903–924](https://github.com/langchain-ai/deepagents/blob/b4ac8829692785a221150658401ae558f4bff3f3/libs/deepagents/deepagents/graph.py#L903-L924), [926–947](https://github.com/langchain-ai/deepagents/blob/b4ac8829692785a221150658401ae558f4bff3f3/libs/deepagents/deepagents/graph.py#L926-L947).

### C13

[langchain-ai/deepagents — libs/deepagents/deepagents/middleware/filesystem.py](https://github.com/langchain-ai/deepagents/blob/b4ac8829692785a221150658401ae558f4bff3f3/libs/deepagents/deepagents/middleware/filesystem.py)

Версия: `b4ac8829692785a221150658401ae558f4bff3f3`; строк: 3881. Git blob: `1ac7756f553ce47bdc27c9ee99203f571741c071`. Опорные фрагменты: [1608–1629](https://github.com/langchain-ai/deepagents/blob/b4ac8829692785a221150658401ae558f4bff3f3/libs/deepagents/deepagents/middleware/filesystem.py#L1608-L1629), [1819–1840](https://github.com/langchain-ai/deepagents/blob/b4ac8829692785a221150658401ae558f4bff3f3/libs/deepagents/deepagents/middleware/filesystem.py#L1819-L1840), [1870–1891](https://github.com/langchain-ai/deepagents/blob/b4ac8829692785a221150658401ae558f4bff3f3/libs/deepagents/deepagents/middleware/filesystem.py#L1870-L1891), [1876–1897](https://github.com/langchain-ai/deepagents/blob/b4ac8829692785a221150658401ae558f4bff3f3/libs/deepagents/deepagents/middleware/filesystem.py#L1876-L1897), [1882–1903](https://github.com/langchain-ai/deepagents/blob/b4ac8829692785a221150658401ae558f4bff3f3/libs/deepagents/deepagents/middleware/filesystem.py#L1882-L1903).

### C14

[langchain-ai/deepagents — libs/deepagents/deepagents/middleware/subagents.py](https://github.com/langchain-ai/deepagents/blob/b4ac8829692785a221150658401ae558f4bff3f3/libs/deepagents/deepagents/middleware/subagents.py)

Версия: `b4ac8829692785a221150658401ae558f4bff3f3`; строк: 1061. Git blob: `2eb9b4279918dc09f33cb6e974f4c4863b72772b`. Опорные фрагменты: [149–170](https://github.com/langchain-ai/deepagents/blob/b4ac8829692785a221150658401ae558f4bff3f3/libs/deepagents/deepagents/middleware/subagents.py#L149-L170), [914–935](https://github.com/langchain-ai/deepagents/blob/b4ac8829692785a221150658401ae558f4bff3f3/libs/deepagents/deepagents/middleware/subagents.py#L914-L935), [953–974](https://github.com/langchain-ai/deepagents/blob/b4ac8829692785a221150658401ae558f4bff3f3/libs/deepagents/deepagents/middleware/subagents.py#L953-L974).

### C15

[microsoft/RD-Agent — rdagent/app/data_science/loop.py](https://github.com/microsoft/RD-Agent/blob/484776c211e4fbbeef03e0ec00d6bbee7362a4f4/rdagent/app/data_science/loop.py)

Версия: `484776c211e4fbbeef03e0ec00d6bbee7362a4f4`; строк: 84. Git blob: `b2d08abf038aa7d4d3fca28fd2aa23153ba11d89`. Проверены структура и доступные определения; дополнительные фрагменты приведены в тексте исследования.

### C16

[microsoft/RD-Agent — rdagent/scenarios/data_science/proposal/exp_gen/planner/__init__.py](https://github.com/microsoft/RD-Agent/blob/484776c211e4fbbeef03e0ec00d6bbee7362a4f4/rdagent/scenarios/data_science/proposal/exp_gen/planner/__init__.py)

Версия: `484776c211e4fbbeef03e0ec00d6bbee7362a4f4`; строк: 46. Git blob: `82559267965e5ab413e3c8fd764996a437191487`. Опорные фрагменты: [27–46](https://github.com/microsoft/RD-Agent/blob/484776c211e4fbbeef03e0ec00d6bbee7362a4f4/rdagent/scenarios/data_science/proposal/exp_gen/planner/__init__.py#L27-L46).

### C17

[SakanaAI/AI-Scientist-v2 — ai_scientist/treesearch/agent_manager.py](https://github.com/SakanaAI/AI-Scientist-v2/blob/96bd51617cfdbb494a9fc283af00fe090edfae48/ai_scientist/treesearch/agent_manager.py)

Версия: `96bd51617cfdbb494a9fc283af00fe090edfae48`; строк: 1222. Git blob: `a81a31f9c8e0ecb5a3eae461002f2008e26b096b`. Опорные фрагменты: [200–221](https://github.com/SakanaAI/AI-Scientist-v2/blob/96bd51617cfdbb494a9fc283af00fe090edfae48/ai_scientist/treesearch/agent_manager.py#L200-L221), [410–431](https://github.com/SakanaAI/AI-Scientist-v2/blob/96bd51617cfdbb494a9fc283af00fe090edfae48/ai_scientist/treesearch/agent_manager.py#L410-L431), [692–713](https://github.com/SakanaAI/AI-Scientist-v2/blob/96bd51617cfdbb494a9fc283af00fe090edfae48/ai_scientist/treesearch/agent_manager.py#L692-L713).

### C18

[facebookresearch/aira-dojo — src/dojo/solvers/greedy/greedy.py](https://github.com/facebookresearch/aira-dojo/blob/c795d8649d30b77fa9b8c011b978bba7127507ad/src/dojo/solvers/greedy/greedy.py)

Версия: `c795d8649d30b77fa9b8c011b978bba7127507ad`; строк: 550. Git blob: `1e4bfc112a95cd6465c14ba5aacf1425fd4ecf0a`. Опорные фрагменты: [197–218](https://github.com/facebookresearch/aira-dojo/blob/c795d8649d30b77fa9b8c011b978bba7127507ad/src/dojo/solvers/greedy/greedy.py#L197-L218), [242–263](https://github.com/facebookresearch/aira-dojo/blob/c795d8649d30b77fa9b8c011b978bba7127507ad/src/dojo/solvers/greedy/greedy.py#L242-L263), [272–293](https://github.com/facebookresearch/aira-dojo/blob/c795d8649d30b77fa9b8c011b978bba7127507ad/src/dojo/solvers/greedy/greedy.py#L272-L293), [384–405](https://github.com/facebookresearch/aira-dojo/blob/c795d8649d30b77fa9b8c011b978bba7127507ad/src/dojo/solvers/greedy/greedy.py#L384-L405).

### C19

[facebookresearch/aira-dojo — src/dojo/solvers/mcts/mcts.py](https://github.com/facebookresearch/aira-dojo/blob/c795d8649d30b77fa9b8c011b978bba7127507ad/src/dojo/solvers/mcts/mcts.py)

Версия: `c795d8649d30b77fa9b8c011b978bba7127507ad`; строк: 651. Git blob: `d4a541972e6174bf1b1496b57e95cabba49836b6`. Опорные фрагменты: [252–273](https://github.com/facebookresearch/aira-dojo/blob/c795d8649d30b77fa9b8c011b978bba7127507ad/src/dojo/solvers/mcts/mcts.py#L252-L273), [283–304](https://github.com/facebookresearch/aira-dojo/blob/c795d8649d30b77fa9b8c011b978bba7127507ad/src/dojo/solvers/mcts/mcts.py#L283-L304), [309–330](https://github.com/facebookresearch/aira-dojo/blob/c795d8649d30b77fa9b8c011b978bba7127507ad/src/dojo/solvers/mcts/mcts.py#L309-L330), [416–437](https://github.com/facebookresearch/aira-dojo/blob/c795d8649d30b77fa9b8c011b978bba7127507ad/src/dojo/solvers/mcts/mcts.py#L416-L437).

### C20

[facebookresearch/aira-dojo — src/dojo/core/solvers/operators/memory.py](https://github.com/facebookresearch/aira-dojo/blob/c795d8649d30b77fa9b8c011b978bba7127507ad/src/dojo/core/solvers/operators/memory.py)

Версия: `c795d8649d30b77fa9b8c011b978bba7127507ad`; строк: 242. Git blob: `6d8711e46dc187867f4a0bc46e49f39beeb5a04e`. Опорные фрагменты: [225–242](https://github.com/facebookresearch/aira-dojo/blob/c795d8649d30b77fa9b8c011b978bba7127507ad/src/dojo/core/solvers/operators/memory.py#L225-L242).

### C21

[SJTU-SAI-agents/ML-Master — agent/mcts_agent.py](https://github.com/SJTU-SAI-agents/ML-Master/blob/6a7bf550074d09162fa955adcc4ed81017c0e697/agent/mcts_agent.py)

Версия: `6a7bf550074d09162fa955adcc4ed81017c0e697`; строк: 961. Git blob: `6425025b9863143600ba2445dcff2980a4a61631`. Опорные фрагменты: [171–192](https://github.com/SJTU-SAI-agents/ML-Master/blob/6a7bf550074d09162fa955adcc4ed81017c0e697/agent/mcts_agent.py#L171-L192), [239–260](https://github.com/SJTU-SAI-agents/ML-Master/blob/6a7bf550074d09162fa955adcc4ed81017c0e697/agent/mcts_agent.py#L239-L260), [448–469](https://github.com/SJTU-SAI-agents/ML-Master/blob/6a7bf550074d09162fa955adcc4ed81017c0e697/agent/mcts_agent.py#L448-L469), [546–567](https://github.com/SJTU-SAI-agents/ML-Master/blob/6a7bf550074d09162fa955adcc4ed81017c0e697/agent/mcts_agent.py#L546-L567), [676–697](https://github.com/SJTU-SAI-agents/ML-Master/blob/6a7bf550074d09162fa955adcc4ed81017c0e697/agent/mcts_agent.py#L676-L697).

### C22

[stanfordnlp/dspy — dspy/teleprompt/gepa/gepa.py](https://github.com/stanfordnlp/dspy/blob/b4beb23d94711857f618042f4e89260b2e8e09b3/dspy/teleprompt/gepa/gepa.py)

Версия: `b4beb23d94711857f618042f4e89260b2e8e09b3`; строк: 695. Git blob: `c202c0c9d42af7f346acedc2f0bf580441e36514`. Опорные фрагменты: [34–55](https://github.com/stanfordnlp/dspy/blob/b4beb23d94711857f618042f4e89260b2e8e09b3/dspy/teleprompt/gepa/gepa.py#L34-L55), [193–214](https://github.com/stanfordnlp/dspy/blob/b4beb23d94711857f618042f4e89260b2e8e09b3/dspy/teleprompt/gepa/gepa.py#L193-L214), [257–278](https://github.com/stanfordnlp/dspy/blob/b4beb23d94711857f618042f4e89260b2e8e09b3/dspy/teleprompt/gepa/gepa.py#L257-L278), [357–378](https://github.com/stanfordnlp/dspy/blob/b4beb23d94711857f618042f4e89260b2e8e09b3/dspy/teleprompt/gepa/gepa.py#L357-L378), [377–398](https://github.com/stanfordnlp/dspy/blob/b4beb23d94711857f618042f4e89260b2e8e09b3/dspy/teleprompt/gepa/gepa.py#L377-L398).

### C23

[FoundationAgents/MetaGPT — metagpt/ext/sela/search/tree_search.py](https://github.com/FoundationAgents/MetaGPT/blob/11cdf466d042aece04fc6cfd13b28e1a70341b1f/metagpt/ext/sela/search/tree_search.py)

Версия: `11cdf466d042aece04fc6cfd13b28e1a70341b1f`; строк: 493. Git blob: `eac26c86ca7af852bc256023c7de89c34bdb8395`. Проверены структура и доступные определения; дополнительные фрагменты приведены в тексте исследования.

### C24

[FoundationAgents/OpenManus — app/agent/base.py](https://github.com/FoundationAgents/OpenManus/blob/3309bf4e416fb1c74b008f3e86494439a31bad53/app/agent/base.py)

Версия: `3309bf4e416fb1c74b008f3e86494439a31bad53`; строк: 197. Git blob: `65f660073527cee9a9385a482123bd7632de5b09`. Опорные фрагменты: [116–137](https://github.com/FoundationAgents/OpenManus/blob/3309bf4e416fb1c74b008f3e86494439a31bad53/app/agent/base.py#L116-L137), [157–178](https://github.com/FoundationAgents/OpenManus/blob/3309bf4e416fb1c74b008f3e86494439a31bad53/app/agent/base.py#L157-L178), [163–184](https://github.com/FoundationAgents/OpenManus/blob/3309bf4e416fb1c74b008f3e86494439a31bad53/app/agent/base.py#L163-L184), [170–191](https://github.com/FoundationAgents/OpenManus/blob/3309bf4e416fb1c74b008f3e86494439a31bad53/app/agent/base.py#L170-L191).

### C25

[OpenAutoCoder/Agentless — agentless/repair/rerank.py](https://github.com/OpenAutoCoder/Agentless/blob/5ce5888b9f149beaace393957a55ea8ee46c9f71/agentless/repair/rerank.py)

Версия: `5ce5888b9f149beaace393957a55ea8ee46c9f71`; строк: 344. Git blob: `5f1cab553fab0cc369027bd77f404e51bce373a3`. Проверены структура и доступные определения; дополнительные фрагменты приведены в тексте исследования.

### C26

[aaif-goose/goose — crates/goose/src/agents/large_response_handler.rs](https://github.com/aaif-goose/goose/blob/6835dd45a21a701b017abe38aa4204e18fed9213/crates/goose/src/agents/large_response_handler.rs)

Версия: `6835dd45a21a701b017abe38aa4204e18fed9213`; строк: 266. Git blob: `eed34c06bca280d2db65c75a8001efc7e17bd198`. Опорные фрагменты: [14–35](https://github.com/aaif-goose/goose/blob/6835dd45a21a701b017abe38aa4204e18fed9213/crates/goose/src/agents/large_response_handler.rs#L14-L35).

### C27

[Significant-Gravitas/AutoGPT — autogpt_platform/backend/backend/executor/manager.py](https://github.com/Significant-Gravitas/AutoGPT/blob/8d4a965b7bbc87b271cd7e4c163ed09d443e9199/autogpt_platform/backend/backend/executor/manager.py)

Версия: `8d4a965b7bbc87b271cd7e4c163ed09d443e9199`; строк: 2278. Git blob: `388639d9a6baad21d4eaa75f09dc97613799cd87`. Опорные фрагменты: [724–745](https://github.com/Significant-Gravitas/AutoGPT/blob/8d4a965b7bbc87b271cd7e4c163ed09d443e9199/autogpt_platform/backend/backend/executor/manager.py#L724-L745), [1650–1671](https://github.com/Significant-Gravitas/AutoGPT/blob/8d4a965b7bbc87b271cd7e4c163ed09d443e9199/autogpt_platform/backend/backend/executor/manager.py#L1650-L1671).

### C28

[NousResearch/hermes-agent — run_agent.py](https://github.com/NousResearch/hermes-agent/blob/944c3a434071f43e64b443d033f3eba30c64434c/run_agent.py)

Версия: `944c3a434071f43e64b443d033f3eba30c64434c`; строк: 1564. Blob отдельно не запрашивался; ссылка закреплена commit. Проверены структура и доступные определения; дополнительные фрагменты приведены в тексте исследования.

### C29

[letta-ai/letta — README.md](https://github.com/letta-ai/letta/blob/8816d3a8516893bfdc0fa14bfb1235226012e295/README.md)

Версия: `8816d3a8516893bfdc0fa14bfb1235226012e295`; строк: 39. Git blob: `9a23cb4226d462b555ae75052e32370a7d82adee`. Проверены структура и доступные определения; дополнительные фрагменты приведены в тексте исследования.

### C30

[bytedance/trae-agent — trae_agent/agent/trae_agent.py](https://github.com/bytedance/trae-agent/blob/e839e559ac61bdd0e057c375dd1dee391fee797d/trae_agent/agent/trae_agent.py)

Версия: `e839e559ac61bdd0e057c375dd1dee391fee797d`; строк: 259. Git blob: `c414117c7c59ab8e013fc5414209c7105ddc1a06`. Опорные фрагменты: [177–198](https://github.com/bytedance/trae-agent/blob/e839e559ac61bdd0e057c375dd1dee391fee797d/trae_agent/agent/trae_agent.py#L177-L198).

### C31

[microsoft/RD-Agent — rdagent/components/coder/CoSTEER/evolving_strategy.py](https://github.com/microsoft/RD-Agent/blob/484776c211e4fbbeef03e0ec00d6bbee7362a4f4/rdagent/components/coder/CoSTEER/evolving_strategy.py)

Версия: `484776c211e4fbbeef03e0ec00d6bbee7362a4f4`; строк: 173. Git blob: `fbbf3861592591c46ceefa624da401678821e432`. Опорные фрагменты: [108–129](https://github.com/microsoft/RD-Agent/blob/484776c211e4fbbeef03e0ec00d6bbee7362a4f4/rdagent/components/coder/CoSTEER/evolving_strategy.py#L108-L129).

### C32

[crewAIInc/crewAI — lib/crewai/src/crewai/agents/agent_executor.py](https://github.com/crewAIInc/crewAI/blob/f76ee29450c5b142e57f1e9b3909ffc6e416d365/lib/crewai/src/crewai/agents/agent_executor.py)

Версия: `f76ee29450c5b142e57f1e9b3909ffc6e416d365`; строк: 3544. Blob отдельно не запрашивался; ссылка закреплена commit. Опорные фрагменты: [555–576](https://github.com/crewAIInc/crewAI/blob/f76ee29450c5b142e57f1e9b3909ffc6e416d365/lib/crewai/src/crewai/agents/agent_executor.py#L555-L576).

### C33

[crewAIInc/crewAI — lib/crewai/src/crewai/agents/step_executor.py](https://github.com/crewAIInc/crewAI/blob/f76ee29450c5b142e57f1e9b3909ffc6e416d365/lib/crewai/src/crewai/agents/step_executor.py)

Версия: `f76ee29450c5b142e57f1e9b3909ffc6e416d365`; строк: 703. Blob отдельно не запрашивался; ссылка закреплена commit. Опорные фрагменты: [67–88](https://github.com/crewAIInc/crewAI/blob/f76ee29450c5b142e57f1e9b3909ffc6e416d365/lib/crewai/src/crewai/agents/step_executor.py#L67-L88), [251–272](https://github.com/crewAIInc/crewAI/blob/f76ee29450c5b142e57f1e9b3909ffc6e416d365/lib/crewai/src/crewai/agents/step_executor.py#L251-L272).

### C34

[google/adk-recipes — python/agents/machine-learning-engineering/machine_learning_engineering/sub_agents/refinement/agent.py](https://github.com/google/adk-recipes/blob/7d187b6f97522915a153b1cdc30a8f4a164c7b5a/python/agents/machine-learning-engineering/machine_learning_engineering/sub_agents/refinement/agent.py)

Версия: `7d187b6f97522915a153b1cdc30a8f4a164c7b5a`; строк: 494. Git blob: `9595ef58cac4be9378cc63e9c38d40bfc889491e`. Опорные фрагменты: [31–52](https://github.com/google/adk-recipes/blob/7d187b6f97522915a153b1cdc30a8f4a164c7b5a/python/agents/machine-learning-engineering/machine_learning_engineering/sub_agents/refinement/agent.py#L31-L52), [348–369](https://github.com/google/adk-recipes/blob/7d187b6f97522915a153b1cdc30a8f4a164c7b5a/python/agents/machine-learning-engineering/machine_learning_engineering/sub_agents/refinement/agent.py#L348-L369), [488–494](https://github.com/google/adk-recipes/blob/7d187b6f97522915a153b1cdc30a8f4a164c7b5a/python/agents/machine-learning-engineering/machine_learning_engineering/sub_agents/refinement/agent.py#L488-L494).

### C35

[google/adk-recipes — python/agents/machine-learning-engineering/machine_learning_engineering/sub_agents/refinement/prompt.py](https://github.com/google/adk-recipes/blob/7d187b6f97522915a153b1cdc30a8f4a164c7b5a/python/agents/machine-learning-engineering/machine_learning_engineering/sub_agents/refinement/prompt.py)

Версия: `7d187b6f97522915a153b1cdc30a8f4a164c7b5a`; строк: 171. Git blob: `ebf82be9f10459f9b13162e69baee08a5ca9e9d8`. Проверены структура и доступные определения; дополнительные фрагменты приведены в тексте исследования.

### C36

[google/adk-recipes — python/agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/check_leakage_util.py](https://github.com/google/adk-recipes/blob/7d187b6f97522915a153b1cdc30a8f4a164c7b5a/python/agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/check_leakage_util.py)

Версия: `7d187b6f97522915a153b1cdc30a8f4a164c7b5a`; строк: 258. Git blob: `9cd970d3e1385cfc082e4c7b025b64c83b77346c`. Проверены структура и доступные определения; дополнительные фрагменты приведены в тексте исследования.

### C37

[google/adk-recipes — python/agents/machine-learning-engineering/machine_learning_engineering/sub_agents/initialization/agent.py](https://github.com/google/adk-recipes/blob/7d187b6f97522915a153b1cdc30a8f4a164c7b5a/python/agents/machine-learning-engineering/machine_learning_engineering/sub_agents/initialization/agent.py)

Версия: `7d187b6f97522915a153b1cdc30a8f4a164c7b5a`; строк: 496. Git blob: `b1c29bebe8e5de88a849827ca7cd0dd585890c40`. Проверены структура и доступные определения; дополнительные фрагменты приведены в тексте исследования.

### C38

[google/adk-recipes — python/agents/machine-learning-engineering/machine_learning_engineering/sub_agents/ensemble/agent.py](https://github.com/google/adk-recipes/blob/7d187b6f97522915a153b1cdc30a8f4a164c7b5a/python/agents/machine-learning-engineering/machine_learning_engineering/sub_agents/ensemble/agent.py)

Версия: `7d187b6f97522915a153b1cdc30a8f4a164c7b5a`; строк: 255. Git blob: `fba34ca79b3732acc2da7d48aac393cfc3b256b5`. Проверены структура и доступные определения; дополнительные фрагменты приведены в тексте исследования.

### C39

[WecoAI/aideml — aide/agent.py](https://github.com/WecoAI/aideml/blob/ad73d227af481bea3de06f884f41b2993dd6453a/aide/agent.py)

Версия: `ad73d227af481bea3de06f884f41b2993dd6453a`; строк: 440. Git blob: `512368205265b75d5cfdaf777f03db0245e1631e`. Опорные фрагменты: [161–182](https://github.com/WecoAI/aideml/blob/ad73d227af481bea3de06f884f41b2993dd6453a/aide/agent.py#L161-L182), [274–295](https://github.com/WecoAI/aideml/blob/ad73d227af481bea3de06f884f41b2993dd6453a/aide/agent.py#L274-L295), [306–327](https://github.com/WecoAI/aideml/blob/ad73d227af481bea3de06f884f41b2993dd6453a/aide/agent.py#L306-L327), [375–396](https://github.com/WecoAI/aideml/blob/ad73d227af481bea3de06f884f41b2993dd6453a/aide/agent.py#L375-L396), [395–416](https://github.com/WecoAI/aideml/blob/ad73d227af481bea3de06f884f41b2993dd6453a/aide/agent.py#L395-L416).

### C40

[WecoAI/aideml — aide/journal.py](https://github.com/WecoAI/aideml/blob/ad73d227af481bea3de06f884f41b2993dd6453a/aide/journal.py)

Версия: `ad73d227af481bea3de06f884f41b2993dd6453a`; строк: 240. Git blob: `88d44a4e84d014febdbe259ba13bdaf826ebe83e`. Опорные фрагменты: [216–237](https://github.com/WecoAI/aideml/blob/ad73d227af481bea3de06f884f41b2993dd6453a/aide/journal.py#L216-L237).

### C41

[NousResearch/hermes-agent — trajectory_compressor.py](https://github.com/NousResearch/hermes-agent/blob/944c3a434071f43e64b443d033f3eba30c64434c/trajectory_compressor.py)

Версия: `944c3a434071f43e64b443d033f3eba30c64434c`; строк: 869. Blob отдельно не запрашивался; ссылка закреплена commit. Опорные фрагменты: [361–382](https://github.com/NousResearch/hermes-agent/blob/944c3a434071f43e64b443d033f3eba30c64434c/trajectory_compressor.py#L361-L382).

## Локальные основания

[local_inputs.json](local_inputs.json) содержит 23 файла и SHA-256, а [local_snapshot](local_snapshot) — их снимок. Включены действующий код, QA-план, методический проект, отчёт приёмки А, разбор score v022 и разбор реального диагностического v024. Изучение исходного отчёта v024 не включает чтение sealed_control или повторную оценку скрытых меток. Новых обучений и обращений к Qwen нет.

Context7 использован для LangGraph, smolagents, DSPy и отдельного вопроса об Optuna. Его текущие примеры могут относиться к более новым версиям: для будущей реализации приоритет имеют закреплённые зависимости проекта. Параметры TPE/Grid дополнительно проверены непосредственно для Optuna 4.2.1.

