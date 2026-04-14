from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Dict, List

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback when PyYAML unavailable
    yaml = None

from marketplace_eval.agents.agent import Agent
from marketplace_eval.agents.generator_agent import GeneratorAgent
from marketplace_eval.agents.llm_judge_agent import LLMJudgeAgent, create_judge
from marketplace_eval.agents.retriever_agent import (
    RetrieverAgent,
    create_retriever_agent,
)
from marketplace_eval.agents.router_agent import RouterAgent, create_router
from marketplace_eval.core.io_nodes import InputNode, OutputNode
from marketplace_eval.core.link import Link
from marketplace_eval.core.node import Node
from marketplace_eval.humans.profile_generator import generate_profiles
from marketplace_eval.system.simulation_logger import SimulationLogger, StepRecord
from marketplace_eval.system.types import GenerationResult, JudgeFeedback, UserProfile
from marketplace_eval.system.user_population import UserPopulation
from marketplace_eval.system.user_data_utils import (
    load_user_data,
    generate_user_data,
    save_user_data,
)


class System:
    def __init__(self):
        self.input_nodes: Dict[str, Node] = {}
        self.output_nodes: Dict[str, Node] = {}
        self.agents: Dict[str, Agent] = {}
        self.links: List[Link] = []
        self.logger = SimulationLogger()
        self.t = 0
        self.T = 100
        self.users_per_step = 1
        self.total_num_users = 10
        self.population: UserPopulation | None = None
        self.judge_agent_id: str | None = None
        self.config: dict | None = None
        self.generator_ids: List[str] = []
        self.generator_introduce_from: Dict[str, int] = {}

    def initialize_from_config(self, config_path: str | Path):
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as fh:
            raw_text = fh.read()
            if yaml is not None:
                config = yaml.safe_load(raw_text)
            else:
                config = json.loads(raw_text)

        self.config = config

        simulation_cfg = config.get("simulation", {})
        self.T = simulation_cfg.get("horizon", self.T)
        self.users_per_step = simulation_cfg.get("users_per_step", self.users_per_step)
        self.total_num_users = simulation_cfg.get(
            "total_num_users", self.total_num_users
        )
        seed = simulation_cfg.get("seed")

        # First, extract generator node IDs and optional introduce_from from the graph
        generator_ids: List[str] = []
        generator_introduce_from: Dict[str, int] = {}
        for node_cfg in config.get("graph", {}).get("nodes", []):
            if node_cfg["type"].lower() == "generator":
                gid = node_cfg["id"]
                generator_ids.append(gid)
                generator_introduce_from[gid] = node_cfg.get("introduce_from", 0)
        self.generator_ids = generator_ids
        self.generator_introduce_from = generator_introduce_from

        # Generators active from the start (introduce_from == 0) are included in
        # initial user preferences.  Later-introduced generators are added to
        # each profile at their introduction timestep to avoid inflating their
        # preference before they can actually be selected.
        initial_generator_ids = [
            gid for gid in generator_ids if generator_introduce_from[gid] == 0
        ]

        user_cfg = config.get("users", {})
        profiles_cfg = user_cfg.get("profiles", [])
        profiles: List[UserProfile] = []

        # Check if profiles should be auto-generated
        if not profiles_cfg:
            profiles = generate_profiles(
                initial_generator_ids,
                total_num_users=self.total_num_users,
            )
        else:
            # Use manually provided profiles from config.
            # Filter out preferences for generators not yet introduced at t=0.
            initial_set = set(initial_generator_ids)
            for profile_cfg in profiles_cfg:
                preferences = {
                    k: v
                    for k, v in profile_cfg.get("generator_preferences", {}).items()
                    if k in initial_set
                }
                total = sum(preferences.values())
                if total <= 0:
                    raise ValueError(
                        "User preference weights must sum to a positive value"
                    )
                normalised_preferences = {k: v / total for k, v in preferences.items()}
                profile = UserProfile(
                    user_id=profile_cfg["id"],
                    preference_scores=normalised_preferences,
                    question_style=profile_cfg.get("question_style", ""),
                    question_domain=profile_cfg.get("question_domain", ""),
                    sampling_probability=1.0
                    / len(profiles_cfg),  # Equal sampling probability
                    rng_state={},
                )
                profiles.append(profile)

        # Load or generate user data
        user_data_source = user_cfg.get("user_data_source", "")
        user_data_type = user_cfg.get("user_data_type", "q")
        taxonomy_base = user_cfg.get("taxonomy_base", "datamorgana")

        if user_data_source and Path(user_data_source).exists():
            # Load user data from the provided source
            user_data = load_user_data(user_data_source)
        else:
            # Generate user data automatically
            # Note: Document grounding is now determined per question type based on
            # the "needs_documents" field in the taxonomy config
            user_data = generate_user_data(
                taxonomy_base=taxonomy_base,
                user_data_type=user_data_type,
                num_samples=100,  # Default number of samples
                seed=seed,
            )

            # Save generated data to user_data/ directory
            if user_data_source:
                # Use the specified path
                save_path = Path(user_data_source)
            else:
                # Generate a default path
                save_dir = Path("user_data")
                save_dir.mkdir(exist_ok=True)
                filename = f"generated_{taxonomy_base}_{user_data_type}.json"
                save_path = save_dir / filename

            save_user_data(user_data, save_path)
            print(f"Generated user data saved to: {save_path}")

        self.population = UserPopulation(
            profiles=profiles,
            user_data=user_data,
            rng_seed=seed,
        )

        node_map: Dict[str, Node] = {}
        for node_cfg in config.get("graph", {}).get("nodes", []):
            node_type = node_cfg["type"].lower()
            node_id = node_cfg["id"]
            params = node_cfg.get("params", {})
            if node_type == "input":
                node = InputNode(node_id)
                self.add_input_node(node)
            elif node_type == "output":
                node = OutputNode(node_id)
                self.add_output_node(node)
            elif node_type == "generator":
                node = GeneratorAgent(node_id, **params)
                self.add_agent(node)
            elif node_type == "router":
                node = create_router(node_id, **params)
                self.add_agent(node)
            elif node_type == "retriever":
                node = create_retriever_agent(node_id, **params)
                self.add_agent(node)
            else:
                raise ValueError(f"Unsupported node type '{node_type}'")
            node_map[node_id] = node

        # Judges are defined separately from the graph in the top-level
        # "judges" config key — they evaluate outputs but are not graph nodes.
        for judge_cfg in config.get("judges", []):
            judge_id = judge_cfg["id"]
            judge_params = judge_cfg.get("params", {})
            judge_node = create_judge(judge_id, **judge_params)
            self.add_agent(judge_node)
            self.judge_agent_id = judge_id

        for edge_cfg in config.get("graph", {}).get("edges", []):
            source_id = edge_cfg["source"]
            target_id = edge_cfg["target"]
            metadata = edge_cfg.get("metadata")
            source_node = node_map[source_id]
            target_node = node_map[target_id]
            link = Link(source_node, target_node, metadata)
            source_node.add_output_link(link)
            target_node.add_input_link(link)
            self.add_link(link)

        if self.judge_agent_id is None:
            raise ValueError("Configuration must specify at least one judge agent")

    def get_state(self):
        """Method to get the summary of the state of the system."""

        return {
            "t": self.t,
            "T": self.T,
            "total_num_users": self.total_num_users,
            "users_per_step": self.users_per_step,
            "input_nodes": list(self.input_nodes.keys()),
            "output_nodes": list(self.output_nodes.keys()),
            "agents": list(self.agents.keys()),
            "links": [
                f"{link.source_node.node_id}->{link.target_node.node_id}"
                for link in self.links
            ],
        }

    def get_agents(self):
        """Method to get the agents in the system"""

        return self.agents

    def get_human_facing_agents(self) -> List[Agent]:
        """Method to get the human-facing agents in the system

        Human-facing agents are front-end agents that are directly connected to the user_input node.
        They are commonly LLMs.
        """
        human_facing_agents = []

        # Find the user_input node
        user_input_node = self.input_nodes.get("user_input")
        if user_input_node is None:
            return human_facing_agents

        # Get all output links from the user_input node
        for target_id, links in user_input_node.output_links.items():
            # Check if the target node is an agent
            if target_id in self.agents:
                human_facing_agents.append(self.agents[target_id])

        return human_facing_agents

    def get_topology_graph(self, save_path: str | Path | None = None):
        """Method to get the topology graph of the system.

        If ``save_path`` is provided, the graph will be rendered and saved to that
        path (e.g., "topology.png"). Matplotlib is only required when saving.
        """
        import networkx as nx

        G = nx.DiGraph()
        for agent_id in self.agents.keys():
            G.add_node(agent_id)
        for input_node_id in self.input_nodes.keys():
            G.add_node(input_node_id)
        for output_node_id in self.output_nodes.keys():
            G.add_node(output_node_id)
        for link in self.links:
            G.add_edge(link.source_node.node_id, link.target_node.node_id)

        if save_path is not None:
            try:
                import matplotlib.pyplot as plt  # type: ignore
            except ModuleNotFoundError as exc:
                raise RuntimeError("Install matplotlib.") from exc

            pos = nx.spring_layout(G, seed=42)
            plt.figure()
            nx.draw(G, pos, with_labels=True, node_size=800, arrows=True, font_size=10)

            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close()

        return G

    def add_agent(self, agent: Agent):
        """Method to add an agent to the system"""

        self.agents[agent.node_id] = agent

    def add_input_node(self, input_node: Node):
        """Method to add an input node to the system"""

        self.input_nodes[input_node.node_id] = input_node

    def add_output_node(self, output_node: Node):
        """Method to add an output node to the system"""

        self.output_nodes[output_node.node_id] = output_node

    def add_link(self, link: Link):
        """Method to add a link to the system"""

        self.links.append(link)

    def initialize(self):
        """Method to initialize the system"""

        if self.config is None:
            raise RuntimeError("System must be configured before initialization.")

    async def _step(self):
        """Async method to perform a step in the system"""

        if self.population is None or self.judge_agent_id is None:
            raise RuntimeError("System is missing required components.")

        # Introduce generators whose introduce_from matches the current timestep.
        # Each newly-introduced generator is added to every user profile with
        # a fair initial preference (equal to the average of existing models).
        for gid, intro_t in self.generator_introduce_from.items():
            if intro_t == self.t and intro_t > 0:
                for profile in self.population.profiles:
                    profile.add_generator(gid)

        step_records = await self._execute_step()
        self.logger.log_step(step_records)
        self.t += 1

    async def _execute_step(self) -> List[StepRecord]:
        assert self.population is not None
        profiles = self.population.sample_profiles(self.users_per_step)
        tasks = [
            self._run_user_job(idx, profile) for idx, profile in enumerate(profiles)
        ]
        return await asyncio.gather(*tasks)

    async def _run_user_job(self, tau: int, profile: UserProfile) -> StepRecord:
        assert self.population is not None and self.judge_agent_id is not None
        user_query = self.population.sample_question(profile)
        # Only generators introduced by current timestep are selectable
        active_generators = [
            g
            for g in self.generator_ids
            if self.generator_introduce_from.get(g, 0) <= self.t
        ]
        generator_id = profile.choose_generator(
            self.population.rng, available_ids=active_generators or None
        )
        generator = self.agents[generator_id]

        generation: GenerationResult = await generator.invoke(
            user_query=user_query, system=self
        )
        judge_agent: LLMJudgeAgent = self.agents[self.judge_agent_id]  # type: ignore[assignment]
        judge_feedback: JudgeFeedback = await judge_agent.invoke(
            generation=generation, user_query=user_query
        )

        profile.update_preference(generator_id, judge_feedback.score)
        generator.update({"last_score": judge_feedback.score})

        router_ids: List[str] = []
        retriever_ids: List[str] = []
        retrieval_summaries: List[str] = []

        for call in generation.retrievals:
            retriever_ids.append(call.retriever_id)
            retriever = self.agents[call.retriever_id]
            retriever.update(
                {
                    "last_score": judge_feedback.retriever_scores.get(
                        call.retriever_id, 0.0
                    )
                }
            )
            summary = (
                f"{call.retriever_id}: {', '.join(call.documents[:2])}"
                if call.documents
                else call.retriever_id
            )
            retrieval_summaries.append(summary)
            if call.router_id:
                router_ids.append(call.router_id)
                router = self.agents[call.router_id]
                if isinstance(router, RouterAgent):
                    router.update_feedback(
                        call.retriever_id,
                        judge_feedback.retriever_scores.get(call.retriever_id, 0.0),
                    )

        # Serialize the current preference distribution as JSON
        preference_json = json.dumps(profile.preference_scores)

        # Extract judge grade from feedback
        judge_grade = judge_feedback.generator_feedback.get("grade", "")

        record = StepRecord(
            t=self.t,
            tau=tau,
            user_id=profile.user_id,
            qid=user_query.qid,
            user_question=user_query.raw_question,
            ground_truth_answer=user_query.answer,
            generator_id=generation.generator_id,
            generator_response=generation.answer,
            score=judge_feedback.score,
            judge_grade=judge_grade,
            preference_distribution=preference_json,
            router_id=",".join(sorted(set(router_ids))) if router_ids else None,
            router_response=None,
            retriever_id=",".join(retriever_ids) if retriever_ids else None,
            retrieval_result=(
                " || ".join(retrieval_summaries) if retrieval_summaries else None
            ),
        )
        return record

    def run(self, n_steps=None, log_csv_path: str | Path | None = None):
        """
        Method to start the system.
        If n_steps is not provided, the system will run for T steps.

        Args:
            n_steps: Number of steps to run. Defaults to self.T.
            log_csv_path: If provided, simulation logs will be incrementally
                          appended to this CSV file during the run.
        """
        if log_csv_path is not None:
            self.logger.set_csv_path(log_csv_path)

        if n_steps is None:
            n_steps = self.T

        # Run all steps in a single event loop to avoid connection issues
        asyncio.run(self._run_async(n_steps))
        self.stop()

    async def _run_async(self, n_steps: int):
        """Async implementation of run() that executes all steps in one event loop."""
        for step_idx in range(n_steps):
            print(f"Step {step_idx + 1}/{n_steps} (t={self.t})")
            await self._step()

    def stop(self):
        """Method to stop the system"""

        self.config = self.config or {}

    def reset(self):
        """Method to reset the system"""

        self.t = 0
        self.logger = SimulationLogger()
