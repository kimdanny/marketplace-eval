# Evaluation of Agents under Simulated AI Marketplace Dynamics

Official Code Repository for SIGIR 2026 perspective paper: 
"Evaluation of Agents under Simulated AI Marketplace Dynamics"

## Reproduction of Motivating Experiment

This section describes how to reproduce the two main experiments presented in the paper: (DeepSeek V3.2 introduced at t=100) and  (Qwen3 235B introduced at t=100). Both experiments simulate a marketplace with 7 LLM generators where one generator enters mid-simulation to demonstrate market dynamics under late entry.

### Prerequisites

1. Set up API keys for OpenRouter (or your preferred OpenAI-compatible provider):
   ```bash
   export OPENROUTER_API_KEY="your-api-key"
   ```

2. Ensure you have the required dependencies installed (see Installation section below).

### Experiment 1: Qwen late entry

In this experiment, Qwen3 235B enters the market at timestep 100, competing against 6 existing generators.

**Step 1:** The default configuration in `configs/simple_qa_simulation.yaml` is already set for this experiment. Verify that `generator_qwen3_235b_a22b_2507` has `introduce_from: 100` set:

```yaml
{
  "id": "generator_qwen3_235b_a22b_2507",
  "type": "generator",
  "introduce_from": 100,  # Qwen3 enters at t=100
  ...
}
```

**Step 2:** Run the simulation:

```bash
python examples/demo_simple_qa.py --config configs/simple_qa_simulation.yaml \
    --output-dir results/simple_qa/qwen_later_w_kimi \
    --window-interval 10
```

This runs a 200-step simulation (10 users, 5 users/step) and generates:
- `simulation_log.csv`: Complete interaction log
- `market_share.csv` and `market_share.png`: Cumulative market share over time
- `market_share_windows.csv` and `market_share_windows.png`: Market share by 10-step windows
- `market_share_windows_stacked.png`: Stacked area chart

**Step 3:** Generate stacked plots using the post-simulation CLI:

```bash
marketplace-eval-analyze results/simple_qa/qwen_later_w_kimi/simulation_log.csv \
    --output-dir results/simple_qa/qwen_later_w_kimi \
    --window-interval 10 \
    --config configs/simple_qa_simulation.yaml
```

This creates `market_share_windows_stacked.png` with generator introduction markers.

### Experiment 2: deepseek late entry

In this experiment, DeepSeek V3.2 enters the market at timestep 100 instead of Qwen3.

**Step 1:** Modify `configs/simple_qa_simulation.yaml` to move the late introduction from Qwen3 to DeepSeek:

```yaml
# Remove introduce_from from Qwen3:
{
  "id": "generator_qwen3_235b_a22b_2507",
  "type": "generator",
  # introduce_from: 100,  <- Comment out or remove this line
  ...
}

# Add introduce_from to DeepSeek:
{
  "id": "generator_deepseek_v3_2",
  "type": "generator",
  "introduce_from": 100,  # DeepSeek enters at t=100
  ...
}
```

**Step 2:** Run the simulation:

```bash
python examples/demo_simple_qa.py --config configs/simple_qa_simulation.yaml \
    --output-dir results/simple_qa/ds_later_w_kimi \
    --window-interval 10
```

**Step 3:** Compute customer retention rates for both experiments:

```python
from marketplace_eval.post_simulation.crr import compute_crr_report

windows = [(0, 99), (100, 199)]  # Before and after introduction
crr_df = compute_crr_report(
    log_path="results/simple_qa/qwen_later_w_kimi/simulation_log.csv",
    windows=windows
)
print(crr_df)
```

### Key Configuration Parameters

The experiments use the following key settings:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `horizon` | 200 | Total simulation timesteps |
| `total_num_users` | 10 | Number of persistent user profiles |
| `users_per_step` | 5 | Users sampled per timestep |
| `introduce_from` | 100 | Timestep when late generator enters |
| `seed` | 42 | Random seed for reproducibility |

All generators use `temperature: 0.7` and OpenRouter API for model calls. The judge uses GPT-4.1 with `temperature: 0.0` for consistent evaluation.


# General Framework Introduction

**Python version 3.12.11** (Recommended)

This repository provides a prototype platform for simulating multiple interconnected agents competing in an "Agent Marketplace", such as advanced retrieval-augmented generation (RAG) workflows that connect multiple large language model (LLM) generators, routers, and retrievers. The simulation runs over a configurable graph topology and emulates interactions between synthetic users, generation services, retrieval services, and a judging LLM that scores end-to-end answers.

## Project Goals

* Model heterogeneous RAG pipelines as a directed graph of nodes (inputs, generators, routers, retrievers, judges, and outputs).
* Simulate user populations with distinct question styles, domains, and generator preferences.
* Execute asynchronous "jobs" that traverse the graph from user query to judged answer within each simulation step.
* Track rewards for generators and retrievers using judge feedback so that user preferences and routing policies evolve over time.
* Log all interactions for offline analysis or visualization.

## Repository Layout

```
├── src/
│   └── marketplace_eval/       # Installable Python package (pip install marketplace-eval)
│       ├── agents/             # Generator, router, retriever, and judge agent implementations
│       ├── core/               # Node and link primitives plus IO helper nodes
│       ├── humans/             # User taxonomy configurations and automatic profile generation
│       ├── post_simulation/    # Market share, CRR metrics, plotting, and analysis CLI
│       ├── prompts/            # Prompt templates for LLM judges and user simulation
│       ├── system/             # System orchestrator, logging, types, and user population
│       └── utils/              # LLM client utilities for OpenAI-compatible APIs
├── configs/                    # Example YAML simulation configurations
├── examples/                   # Runnable demo scripts (demo_simple_qa.py, demo_sample.py)
├── pyproject.toml              # Package metadata and dependencies
├── requirements.txt            # Full dependency list for development
└── README.md                   # Project documentation (this file)
```

Key modules:

| Module | Description |
| ------ | ----------- |
| `marketplace_eval.agents.agent` | Base `Agent` type shared by all executable nodes; manages async invocation hooks and metadata. |
| `marketplace_eval.agents.generator_agent` | Generator implementation that issues RAG requests, interacts with routers/retrievers, and updates user preference scores. |
| `marketplace_eval.agents.router_agent` | Implements a simple exploration/exploitation router that selects retrievers based on judged performance. |
| `marketplace_eval.agents.retriever_agent` | Provides lightweight embedding-based retrieval backed by in-memory corpora. |
| `marketplace_eval.agents.llm_judge_agent` | Abstract base class for LLM-based judges; subclass it to define custom evaluation logic. |
| `marketplace_eval.core.node` & `marketplace_eval.core.link` | Define graph nodes and directed links, enabling data transmission between agents. |
| `marketplace_eval.humans.profile_generator` | Generates `total_num_users` uniform user profiles with equal sampling probability and equal generator preferences. |
| `marketplace_eval.humans.datamorgana_config` | DataMorgana taxonomy configuration defining question types and user expertise levels. |
| `marketplace_eval.humans.tuna_config` | TUNA (Taxonomy of User Needs and Actions) configuration for more complex user modeling. |
| `marketplace_eval.system.system` | Central orchestrator that builds the graph from configuration, runs simulation steps, and coordinates logging. |
| `marketplace_eval.system.simulation_logger` | Structured logging of step-level interactions (`StepRecord`) with DataFrame/CSV export helpers. |
| `marketplace_eval.system.user_population` | Manages a population of persistent user profiles, samples profiles based on probabilities, and generates questions. |

## Installation

### From PyPI (recommended)

```bash
pip install marketplace-eval
```

For local LLM support via vLLM:
```bash
pip install "marketplace-eval[vllm]"
```

> Some modules are lazy-imported when a class is initialized. Depending on your use case, you may also need:
> ```
> vllm==0.11.0
> beautifulsoup4==4.14.2
> ```

### From source (development)

1. **Create and activate a Python 3.12 environment.**
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install in editable mode.**
   ```bash
   pip install -e .
   ```

   Or install all development dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration File Structure

Simulations are described via a JSON or YAML file (PyYAML is optional; JSON is always supported). The configuration contains three top-level sections:

* `simulation`: global parameters such as horizon length (`horizon`), number of users per step (`users_per_step`), and RNG seed (`seed`).
* `users`: definitions of user profiles and domain-specific question pools. User profiles can be manually specified or automatically generated based on taxonomy configurations.
* `graph`: declarative node and edge definitions that instantiate input/output nodes, generators, routers, retrievers, and judges. Edges define the directed connectivity of the RAG pipeline.

See [`configs/sample_simulation.yaml`](configs/sample_simulation.yaml) for a fully worked example featuring two generators, a router, three retrievers, and a judge along with sample corpora and user profiles.

### User Profile Generation

User profiles can be created in two ways:

1. **Automatic Generation (Recommended)**: Leave the `profiles` field empty. The system will automatically generate `total_num_users` uniform user profiles. The optional `taxonomy_base` field (either `"datamorgana"` or `"tuna"`) governs how the **question pool** is generated — not the profiles themselves.

2. **Manual Specification**: Provide custom profiles in the configuration file with explicit `id`, `question_style`, `question_domain`, and `generator_preferences`.

#### Automatic Profile Generation

When `profiles` is left empty, the system creates `total_num_users` profiles with uniform generator preferences and equal sampling probability (`1 / total_num_users`). The optional `taxonomy_base` setting (e.g. `"datamorgana"`) only affects how the question pool is generated — each question receives independently-sampled taxonomy labels stored in its metadata. Example config:

```yaml
users:
  taxonomy_base: "datamorgana"  # or "tuna"
  user_data_type: "qa"  # "q" (questions only) or "qa" (question-answer pairs)
  user_data_source: "user_data/my_data.json"  # Path to user data file (optional)
  profiles: []  # Leave empty for auto-generation
```

**User Data Configuration**:
- `user_data_type`: Specifies whether the data contains just questions (`"q"`) or question-answer pairs (`"qa"`). When `"qa"`, reference-based answer evaluation becomes possible.
- `user_data_source`: Path to a JSON file containing user data. If the file exists, it will be loaded. If empty or the file doesn't exist, user data will be automatically generated and saved to the `user_data/` directory.

**Document Grounding**: Whether questions are grounded by documents is determined automatically based on the `needs_documents` field defined for each question type in the taxonomy configuration files (`humans/datamorgana_config.py` or `humans/tuna_config.py`). When a profile's question types require documents, the system automatically samples documents from the corpus to ground the generated questions. Generated data with document-grounded questions will include `context` and `document_ids` fields, enabling reference-based retrieval evaluation.

The `taxonomy_base` field controls how the **question pool** is generated, not the user profiles. When set to `"datamorgana"`, each generated question is assigned taxonomy labels by independently sampling one value per dimension according to its probability distribution:

**DataMorgana Taxonomy** (based on [DataMorgana: Generating Q&A Benchmarks for RAG Evaluation](https://aclanthology.org/2025.acl-industry.33/)):
- **Factuality**: factoid (70%), open-ended (30%)
- **Premise**: without-premise (80%), with-premise (20%)
- **Phrasing**: concise-and-natural (25%), verbose-and-natural (25%), short-search-query (25%), long-search-query (25%)
- **Linguistic Variation**: similar-to-document (50%), distant-from-document (50%)
- **User Expertise**: expert (40%), novice (60%)

These dimensions are sampled **independently per generated question** and stored in the question's metadata.

Generator preferences are initialized with equal probability across all generator nodes in the graph (e.g., 0.5 each if there are two generators).

**Profile Persistence:** Each profile is a persistent user that maintains state across the simulation. When profiles are sampled at each step, they are references to the same objects, so any learned preferences persist automatically. The same profile can be sampled multiple times, simulating returning users who have learned which generators work best for their needs.

#### Manual Profile Specification

To manually define profiles:

```yaml
users:
  profiles:
    - id: "tech_formal"
      question_style: "formal"
      question_domain: "technology"
      generator_preferences:
        generator_qwen_2_5_7b_instruct: 0.6
        generator_gpt_oss_20b: 0.4
```

Each manual profile gets equal sampling probability by default (1/N where N is the number of profiles).

### Adding Nodes

Each node entry includes an `id`, `type`, and optional `params` block. Supported types are:

* `input` / `output`
* `generator`
* `router`
* `retriever`
* `judge`

Generator, router, retriever, and judge nodes accept additional parameters that match their constructor signatures (e.g., model configuration, exploration rates, corpora, or scoring noise).

#### Configuring generator models

Generator nodes can now target either a locally hosted HuggingFace/vLLM model or any OpenAI-compatible HTTP API such as OpenRouter or OpenAI itself. Provide the configuration via the `model` block inside the node `params`:

```yaml
params:
  name: "T5-Base"
  system_prompt: "You are T5 agent, an expert assistant."
  model:
    provider: vllm
    model_id: google-t5/t5-base
    sampling_parameters:
      temperature: 0.7
      max_tokens: 512
```

For OpenRouter or OpenAI, the `provider` should be `openrouter` or `openai` respectively. These providers share an OpenAI-compatible API surface:

```yaml
params:
  name: "GPT-OSS 20B"
  system_prompt: "Answer questions clearly and cite retrieved evidence."
  model:
    provider: openrouter
    model_id: openrouter/gpt-4.1-mini
    base_url: https://openrouter.ai/api/v1   # optional override
    generation_parameters:
      temperature: 0.6
      max_tokens: 512
    headers:
      HTTP-Referer: https://example.com/simulation
      X-Title: Multifirm Simulation
```

If you omit the `model` block, `_llm_client` is not initialized and the generator falls back to echoing the rendered prompt as the answer. To enable real LLM calls ensure the relevant dependencies are installed (`vllm` for HuggingFace deployments, `httpx` for OpenRouter/OpenAI) and set the appropriate API keys via environment variables (e.g., `OPENROUTER_API_KEY` or `OPENAI_API_KEY`).

#### Configuring generator retrieval strategies

Every generator node may be paired with a `retrieval_strategy` configuration that controls whether and how the model calls routers or retrievers. Two strategies are currently supported:

* `naive` (default): the generator calls every connected router and direct retriever exactly once.
* `agentic`: the generator evaluates the user query and then decides which routers or retrievers to call, how many rounds to run, and when to stop based on configurable heuristics.

Common parameters accepted by both strategies include:

* `router_top_k`: limit for how many retrievers a router should return per round (set to `null` to let the router decide dynamically).
* `documents_per_retriever`: number of documents taken from each retrieval call when building the prompt context (set to `null` to keep everything).
* `max_context_documents`: overall cap on how many documents are concatenated into the prompt context.
* `context_separator`: string used to join retrieved snippets inside the prompt template.

The `agentic` strategy accepts additional knobs:

* `max_rounds`: maximum number of retrieval rounds to perform. Each round may invoke routers and direct retrievers.
* `max_retrieval_calls`: optional global cap on the total number of retrieval calls made (the same retriever can count more than once).
* `target_num_documents`: stop early once this many documents have been retrieved.

Example:

```yaml
params:
  name: "Agentic Generator"
  system_prompt: "Reason about whether additional evidence is needed before answering."
  retrieval_strategy:
    type: agentic
    max_rounds: 3
    router_top_k: null            # let the router decide
    max_retrieval_calls: 4
    target_num_documents: 6
    documents_per_retriever: 2
    max_context_documents: 4
```

Setting `type: naive` while leaving other fields blank reproduces the legacy "always retrieve" behaviour, which is helpful for baselines and ablation studies.

#### Configuring retrievers

Retriever nodes default to the FAISS-based index search implemented by `RetrieverAgent`. To connect a retriever to Google Search instead, set the `backend` parameter to `google_search` and provide the corresponding params:

```yaml
params:
  "backend": "google_search",
  "safe": "active",
  "top_k": 3
  "fetch_full_text": True,
  "max_text_length": 1000,
  "url_timeout": 5.0
```

The `google_search` backend returns the same payload structure as the corpus-backed agent (`{"documents": [...], "scores": [...], "query": ...}`), allowing it to plug into existing pipelines without modification. Network errors or invalid credentials result in an empty document list so downstream nodes can handle the failure gracefully.

> **Note:** An `exa_search` backend is not yet implemented. Passing `backend: exa_search` raises `NotImplementedError`.

#### Configuring judges (custom evaluation)

`LLMJudgeAgent` is an **abstract base class** that handles the LLM plumbing (client setup, async invocation, error handling). To run a simulation you must provide a concrete judge by subclassing it and registering it with the `@register_judge` decorator. The config's `judge_prompt` field is the key that maps to your registered class.

**Step 1 — declare the judge in your config:**

```yaml
"judges": [
  {
    "id": "my_judge",
    "type": "judge",
    "params": {
      "judge_prompt": "my_eval",         # <-- must match @register_judge key
      "provider": "openrouter",
      "model_id": "openai/gpt-4.1",
      "generation_parameters": { "temperature": 0.0 }
    }
  }
]
```

**Step 2 — define and register the judge in your demo / entry-point script:**

```python
from marketplace_eval.agents.llm_judge_agent import LLMJudgeAgent, register_judge
from marketplace_eval.system.types import GenerationResult, JudgeFeedback, UserQuery

@register_judge("my_eval")
class MyJudgeAgent(LLMJudgeAgent):

    def format_prompt(self, *, generation, user_query=None):
        """Build the evaluation prompt sent to the LLM."""
        ground_truth = (user_query.answer if user_query else None) or ""
        return (
            f"Question: {generation.question}\n"
            f"Ground-truth: {ground_truth}\n"
            f"Answer: {generation.answer}\n\n"
            "Rate the answer 0-10. Respond with ONLY an integer."
        )

    def parse_llm_response(self, response):
        """Extract a numeric score from the LLM response."""
        for token in response.strip().split():
            try:
                v = int(token)
                if 0 <= v <= 10:
                    return v / 5.0 - 1.0   # normalise to [-1, 1]
            except ValueError:
                continue
        return -1.0

    # Optional: override build_judge_feedback to attach rationale,
    # retriever scores, or any experiment-specific metadata.
    def build_judge_feedback(self, *, score, raw_response, generation, user_query=None):
        retriever_scores = {}
        for call in generation.retrievals:
            retriever_scores[call.retriever_id] = (
                sum(call.scores) / len(call.scores) if call.scores else 0.0
            )
        return JudgeFeedback(
            score=score,
            rationale=f"Raw: {raw_response.strip()[:100]}",
            retriever_scores=retriever_scores,
            generator_feedback={"raw_response": raw_response.strip()},
        )
```

The three methods you can implement are:

| Method | Required? | Purpose |
|--------|-----------|---------|
| `format_prompt` | **Yes** | Build the full prompt string from the generation result and (optionally) the original user query. |
| `parse_llm_response` | **Yes** | Extract a single numeric score (`float`) from the raw LLM output. |
| `build_judge_feedback` | No | Construct the full `JudgeFeedback` object. The default returns minimal feedback with only the score; override to add a rationale, retriever scores, generator-specific metadata, etc. |

See `examples/demo_simple_qa.py` for a complete SimpleQA factuality judge and `examples/demo_sample.py` for a minimal 0-10 scoring judge.

### Defining Edges

Edges connect node IDs and may optionally include metadata. During initialization the system attaches each edge as a `Link` from the source node to the target node so data can flow through the pipeline during simulation.

### Customizing User Behaviour

The simulation's user-side logic is driven by two classes that can be subclassed to change behaviour without modifying the framework internals:

| Class | Overridable Methods | Default Behaviour |
|-------|-------------------|-------------------|
| `UserProfile` | `choose_generator`, `update_preference` | Epsilon-greedy generator selection; linear drift preference update |
| `UserPopulation` | `sample_profiles` | Weighted random sampling with replacement |

Because `System` calls these methods on the profile/population objects directly, Python's method resolution dispatches to your overrides automatically — no changes to the simulation engine are required.

#### Overriding `UserProfile`

Subclass `UserProfile` and override `choose_generator` and/or `update_preference` to implement a custom selection or learning strategy. For example, a softmax-based profile:

```python
from dataclasses import dataclass
from marketplace_eval.system.types import UserProfile

@dataclass
class SoftmaxUserProfile(UserProfile):
    temperature: float = 0.5
    ema_alpha: float = 0.3

    def choose_generator(self, rng, epsilon=0.2, available_ids=None):
        # Softmax sampling over preference scores instead of epsilon-greedy
        import math
        generators = list(self.preference_scores.keys())
        if available_ids is not None:
            generators = [g for g in generators if g in available_ids] or list(available_ids)
        scores = [self.preference_scores.get(g, 0.0) for g in generators]
        max_s = max(scores)
        exp_scores = [math.exp((s - max_s) / max(self.temperature, 1e-8)) for s in scores]
        total = sum(exp_scores)
        return rng.choices(generators, weights=[e / total for e in exp_scores], k=1)[0]

    def update_preference(self, generator_id, score, drift_rate=0.1):
        # Exponential moving average instead of linear drift
        normalised = (score + 1.0) / 2.0
        old = self.preference_scores.get(generator_id, 0.01)
        self.preference_scores[generator_id] = self.ema_alpha * normalised + (1 - self.ema_alpha) * old
        total = sum(self.preference_scores.values())
        if total > 0:
            for k in self.preference_scores:
                self.preference_scores[k] /= total
```

#### Overriding `UserPopulation`

Subclass `UserPopulation` and override `sample_profiles` to change how users are sampled at each timestep. For example, a round-robin sampler that guarantees equal coverage:

```python
from marketplace_eval.system.user_population import UserPopulation

class RoundRobinPopulation(UserPopulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._rr_index = 0

    def sample_profiles(self, n):
        sampled = []
        for _ in range(n):
            sampled.append(self.profiles[self._rr_index % len(self.profiles)])
            self._rr_index += 1
        return sampled
```

#### Injecting Custom Classes

After loading the config, replace the population on the `System` instance before calling `run()`:

```python
system = System()
system.initialize_from_config("configs/sample_simulation.yaml")

custom_profiles = [SoftmaxUserProfile(...) for _ in range(num_users)]
system.population = RoundRobinPopulation(
    profiles=custom_profiles,
    user_data=system.population.user_data,
    rng_seed=system.config.get("simulation", {}).get("seed"),
)

system.initialize()
system.run()
```

See `examples/demo_sample.py` for a complete working example that combines all three overrides (`choose_generator`, `update_preference`, and `sample_profiles`).

## Running the Simulation

The snippet below demonstrates how to bootstrap the system, load the sample configuration, and run a short simulation horizon:

```python
from marketplace_eval.system.system import System

system = System()
system.initialize_from_config("configs/sample_simulation.yaml")
state = system.get_state()
print("Graph nodes:", state["agents"])

system.run()  # handles the event loop internally

print("Completed steps:", system.t)
for record in system.logger.step_records:
    print(record)
```

Each call to `run()` executes asynchronous jobs for the configured number of time steps. Within a step the system:

1. Samples user profiles from the population based on their sampling probabilities. Profiles are persistent objects, so the same user may be sampled multiple times across steps.
2. Samples questions for each profile via `UserPopulation`.
3. Routes each user to a generator according to their current preference distribution.
4. Allows the generator to call routers/retrievers (or skip routers) based on the graph topology.
5. Aggregates retrieval results to craft an answer.
6. Uses the judge to score retrieval and generation quality.
7. Updates the profile's generator preference weights and router statistics using the judge feedback. These updates persist across steps since profiles are mutable objects.
8. Logs the interaction as a `StepRecord`.

## Inspecting Logs

`SimulationLogger` stores every step record and can export the data:

```python
from marketplace_eval.system.simulation_logger import SimulationLogger

logger = system.logger
df = logger.to_df()
print(df.head())
logger.to_csv("simulation_log.csv")
```

## Post-Simulation Analysis

After running a simulation, the `post_simulation/` directory provides tools for analyzing market dynamics and agent performance. These tools read the simulation logs and generate metrics and visualizations.

### Market Share Analysis

The `market_share.py` module computes and visualizes how generator market share evolves over time:

```python
from marketplace_eval.post_simulation.market_share import (
    compute_market_share,
    compute_market_share_windows,
    get_windows_from_interval,
)
from marketplace_eval.post_simulation.plot import (
    plot_market_share,
    plot_market_share_windows_stacked,
)

market_share = compute_market_share("simulation_log.csv")
windows = get_windows_from_interval("simulation_log.csv", interval=20)
ms_windows = compute_market_share_windows("simulation_log.csv", windows)
plot_market_share_windows_stacked(ms_windows, save_path="results/market_share_stacked.png")
```

**Key features:**

* **Cumulative market share**: Tracks each generator's share from simulation start to end
* **Windowed market share**: Computes shares over fixed time windows (e.g., steps 1-20, 21-40)
* **Late introduction tracking**: Automatically masks pre-introduction periods when generators enter mid-simulation
* **Multiple visualizations**: Line plots, windowed progression, and stacked area charts

**CLI usage:**

```bash
marketplace-eval-analyze simulation_log.csv \
    --output-dir results \
    --window-interval 20 \
    --config configs/sample_simulation.yaml
```

The `--config` flag reads `introduce_from` timestamps from the simulation configuration to properly visualize late-entering generators.

**Output files:**

* `market_share.csv`: Cumulative market share at each timestep
* `market_share.png`: Line plot of market share progression
* `market_share_windows.csv`: Market share by time window
* `market_share_windows.png`: Windowed market share line plot
* `market_share_windows_stacked.png`: Stacked area chart showing share evolution

### Customer Retention Rate (CRR)

The `crr.py` module measures how well generators retain users after initial adoption. For each generator and user, it computes the fraction of subsequent interactions where the user continues selecting that generator:

```python
from marketplace_eval.post_simulation.crr import compute_crr_report

# Compute retention rates for two windows
windows = [(0, 99), (100, 199)]  # Steps 1-100 and 101-200
crr_df = compute_crr_report(
    log_path="simulation_log.csv",
    windows=windows,
)
print(crr_df)
```

**Output:**

* DataFrame with rows = time windows, columns = generators, values = retention rates (0-1 scale)

