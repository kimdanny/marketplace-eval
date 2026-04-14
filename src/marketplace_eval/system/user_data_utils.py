"""Utilities for loading and generating user data for simulations."""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import Dict, List, Any

from marketplace_eval.humans.datamorgana_config import DM_QUESTION_TYPES, DM_USER_TYPES
from marketplace_eval.prompts.user_simulation_prompt import (
    DM_QA_GENERATION_PROMPT_DOCUMENT_BASED,
    DM_QUESTION_GENERATION_PROMPT_DOCUMENT_BASED,
)
from marketplace_eval.utils.llm_client import create_llm_client


def load_user_data(file_path: str | Path) -> List[Dict[str, Any]]:
    """
    Load user data from a JSON, JSONL, or CSV file.

    For CSV files, supports the simple_qa format with columns:
    - qid: question identifier
    - metadata: (ignored)
    - problem: the question text (mapped to "question")
    - answer: ground truth answer

    Args:
        file_path: Path to the user data file

    Returns:
        List of user data entries, each containing at minimum a "question" field
        and optionally "qid", "answer", "context", and "document_ids" fields
    """
    import csv

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"User data file not found: {file_path}")

    if path.suffix == ".csv":
        # Load CSV format (e.g., simple_qa with columns: qid, metadata, problem, answer)
        data = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entry: Dict[str, Any] = {}
                # Map "problem" column to "question"
                if "problem" in row:
                    entry["question"] = row["problem"]
                elif "question" in row:
                    entry["question"] = row["question"]
                else:
                    raise ValueError(f"CSV must have a 'problem' or 'question' column")
                # Map qid
                if "qid" in row:
                    entry["qid"] = row["qid"]
                # Map answer
                if "answer" in row:
                    entry["answer"] = row["answer"]
                # Ignore metadata column
                data.append(entry)
    else:
        with path.open("r", encoding="utf-8") as f:
            if path.suffix == ".jsonl":
                # Load JSONL format (one JSON object per line)
                data = [json.loads(line) for line in f if line.strip()]
            else:
                # Load regular JSON format
                data = json.load(f)

    # Validate data format
    if not isinstance(data, list):
        raise ValueError("User data must be a list of objects")

    for idx, entry in enumerate(data):
        if "question" not in entry:
            raise ValueError(f"Entry {idx} is missing required 'question' field")

    return data


def _sample_taxonomy_type(type_list: list, rng: random.Random) -> dict:
    """Sample one entry from a taxonomy type list using probability weights."""
    names = [t["name"] for t in type_list]
    weights = [t["probability"] for t in type_list]
    chosen_name = rng.choices(names, weights=weights, k=1)[0]
    return next(t for t in type_list if t["name"] == chosen_name)


def _sampled_types_need_documents(
    sampled_question_types: Dict[str, dict],
) -> bool:
    """Check if any of the sampled question types require documents."""
    for type_info in sampled_question_types.values():
        if type_info.get("needs_documents", False):
            return True
    return False


def generate_user_data(
    taxonomy_base: str,
    user_data_type: str,
    num_samples: int = 100,
    seed: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Generate synthetic user data for simulation using LLM.

    Question types and user types are sampled independently per instance from
    the taxonomy's probability distributions.  This decouples question
    characteristics from user profiles.

    Args:
        taxonomy_base: Either "datamorgana" or "tuna"
        user_data_type: Either "q" (questions only) or "qa" (question-answer pairs)
        num_samples: Number of data entries to generate
        seed: Random seed for reproducibility

    Returns:
        List of generated user data entries, each with a user_id field
    """
    return asyncio.run(
        _generate_user_data_async(
            taxonomy_base=taxonomy_base,
            user_data_type=user_data_type,
            num_samples=num_samples,
            seed=seed,
        )
    )


async def _generate_user_data_async(
    taxonomy_base: str,
    user_data_type: str,
    num_samples: int,
    seed: int | None,
) -> List[Dict[str, Any]]:
    """Async implementation of user data generation."""
    rng = random.Random(seed)

    if taxonomy_base.lower() != "datamorgana":
        raise NotImplementedError(
            f"LLM-based user data generation is only implemented for 'datamorgana', "
            f"not '{taxonomy_base}'"
        )

    # Load documents for question types that need document grounding
    documents = _load_fineweb_documents("user_data/fineweb_500.jsonl")
    if not documents:
        print(
            "Warning: No documents found in fineweb sample file. "
            "Document-grounded question types may fail."
        )

    # Initialize LLM client
    llm_client = create_llm_client(
        {
            "provider": "openai",
            "model_id": "gpt-4o",
            "generation_parameters": {
                "temperature": 0.7,
                "max_tokens": 512,
            },
        }
    )
    if llm_client is None:
        raise ValueError("Failed to create LLM client")

    if user_data_type == "qa":
        prompt_template = DM_QA_GENERATION_PROMPT_DOCUMENT_BASED
    elif user_data_type == "q":
        prompt_template = DM_QUESTION_GENERATION_PROMPT_DOCUMENT_BASED
    else:
        raise ValueError(
            f"Invalid user_data_type: {user_data_type}. Must be 'q' or 'qa'"
        )

    # Pre-sample all taxonomy types deterministically before async work
    sampled_question_types_list = [
        {
            cat: _sample_taxonomy_type(types, rng)
            for cat, types in DM_QUESTION_TYPES.items()
        }
        for _ in range(num_samples)
    ]
    sampled_user_types_list = [
        {cat: _sample_taxonomy_type(types, rng) for cat, types in DM_USER_TYPES.items()}
        for _ in range(num_samples)
    ]

    tasks = []
    for i in range(num_samples):
        sampled_q_types = sampled_question_types_list[i]
        sampled_u_types = sampled_user_types_list[i]

        needs_documents = _sampled_types_need_documents(sampled_q_types)
        document = None
        document_id = None
        if needs_documents and documents:
            doc_entry = rng.choice(documents)
            document = doc_entry["text"]
            document_id = doc_entry["id"]
        elif needs_documents and not documents:
            print(
                f"Warning: Instance {i} needs documents but none are available. "
                "Skipping..."
            )
            continue

        task = _generate_single_entry(
            llm_client=llm_client,
            prompt_template=prompt_template,
            sampled_question_types=sampled_q_types,
            sampled_user_types=sampled_u_types,
            user_data_type=user_data_type,
            document=document,
            document_id=document_id,
            instance_index=i,
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    data: List[Dict[str, Any]] = []
    for result in results:
        if isinstance(result, dict):
            data.append(result)
        elif isinstance(result, Exception):
            print(f"Warning: Generation failed: {result}")

    return data[:num_samples]


def _load_fineweb_documents(file_path: str) -> List[Dict[str, str]]:
    """Load documents from fineweb JSONL file."""
    documents = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    doc = json.loads(line)
                    if "text" in doc and "id" in doc:
                        documents.append({"text": doc["text"], "id": doc["id"]})
    except FileNotFoundError:
        print(f"Warning: Fineweb file not found at {file_path}")
    except Exception as e:
        print(f"Warning: Error loading fineweb documents: {e}")

    return documents


async def _generate_single_entry(
    llm_client,
    prompt_template: str,
    sampled_question_types: Dict[str, dict],
    sampled_user_types: Dict[str, dict],
    user_data_type: str,
    document: str | None,
    document_id: str | None,
    instance_index: int,
) -> Dict[str, Any]:
    """Generate a single user data entry using LLM."""
    formatted_prompt = _format_prompt(
        prompt_template=prompt_template,
        sampled_question_types=sampled_question_types,
        sampled_user_types=sampled_user_types,
        document=document,
    )

    try:
        response = await llm_client.generate(formatted_prompt)
        entry = _parse_llm_response(response, user_data_type)

        if document is not None:
            entry["context"] = [document]
            if document_id:
                entry["document_ids"] = [document_id]

        entry["metadata"] = {
            "question_types": {
                cat: t["name"] for cat, t in sampled_question_types.items()
            },
            "user_types": {cat: t["name"] for cat, t in sampled_user_types.items()},
        }

        return entry

    except Exception as e:
        raise RuntimeError(f"Failed to generate entry (instance {instance_index}): {e}")


def _format_prompt(
    prompt_template: str,
    sampled_question_types: Dict[str, dict],
    sampled_user_types: Dict[str, dict],
    document: str | None,
) -> str:
    """Format the generation prompt with independently sampled taxonomy types."""
    prompt_kwargs: Dict[str, str] = {}

    if document:
        prompt_kwargs["document"] = document

    for cat, type_info in sampled_question_types.items():
        key = f"question_type_description_{cat.replace('-', '_')}"
        prompt_kwargs[key] = f"- {type_info['description']}"

    for cat, type_info in sampled_user_types.items():
        key = f"user_type_description_{cat.replace('-', '_')}"
        prompt_kwargs[key] = f"The user is {type_info['description']}"

    formatted = prompt_template
    for key, value in prompt_kwargs.items():
        placeholder = "{" + key + "}"
        formatted = formatted.replace(placeholder, value)

    return formatted


def _parse_llm_response(response: str, user_data_type: str) -> Dict[str, Any]:
    """Parse LLM response into structured data."""
    if user_data_type == "qa":
        # Try to parse JSON format: {"question": "...", "answer": "..."}
        try:
            response = response.strip()
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            data = json.loads(response)
            if "question" in data and "answer" in data:
                return {"question": data["question"], "answer": data["answer"]}
        except json.JSONDecodeError:
            lines = response.strip().split("\n")
            question = None
            answer = None

            for line in lines:
                if line.strip().startswith('"question"'):
                    question = line.split(":", 1)[1].strip(' ",')
                elif line.strip().startswith('"answer"'):
                    answer = line.split(":", 1)[1].strip(' ",')

            if question and answer:
                return {"question": question, "answer": answer}

            if "\n\n" in response:
                parts = response.split("\n\n", 1)
                return {"question": parts[0].strip(), "answer": parts[1].strip()}
            else:
                return {"question": response.strip(), "answer": ""}

    else:  # user_data_type == "q"
        response = response.strip()
        if "```" in response:
            response = response.split("```")[1].split("```")[0].strip()

        return {"question": response}


def save_user_data(data: List[Dict[str, Any]], file_path: str | Path):
    """
    Save user data to a JSON file.

    Args:
        data: List of user data entries
        file_path: Path where to save the data
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
