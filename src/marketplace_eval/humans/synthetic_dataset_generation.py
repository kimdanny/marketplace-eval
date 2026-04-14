"""
Based on the taxonomy configuration, generate a dataset.
Dataset can be either a question-answer pair or a question only.
Generated dataset will be saved under the user_data/ directory.
"""

import asyncio
import csv
import json
import random

from marketplace_eval.prompts.user_simulation_prompt import (
    DM_QA_GENERATION_PROMPT_DOCUMENT_BASED,
    DM_QUESTION_GENERATION_PROMPT_DOCUMENT_BASED,
)
from marketplace_eval.utils.llm_client import BaseLLMClient
from marketplace_eval.humans.datamorgana_config import DM_QUESTION_TYPES, DM_USER_TYPES


def _sample_by_probability(type_list: list, rng: random.Random) -> dict:
    """Sample one entry from type_list using the 'probability' field as weights."""
    names = [t["name"] for t in type_list]
    weights = [t["probability"] for t in type_list]
    chosen_name = rng.choices(names, weights=weights, k=1)[0]
    return next(t for t in type_list if t["name"] == chosen_name)


def generate_dataset_document_based(generation_config: dict):
    """
    Generate a dataset based on the taxonomy configuration and sampled documents.
    Dataset can be either a question-answer pair or a question only.
    Args:
        generation_config: Configuration for dataset generation.
            - taxonomy_base: "datamorgana"
            - dataset_type: "qa" or "q"
            - num_instances: Number of instances to generate
            - seed: Random seed for reproducibility
            - generator_llm_client: LLM client for dataset generation (type: BaseLLMClient)
            - document_sampling_pool_path: Path to the document sampling pool
                Example line of document sampling pool dataset in jsonl format (e.g., source: fineweb).
                {"text": "|Viewing Single Post From: Spoilers for ...", "id": "<urn:uuid:39147604-bfbe-4ed5-b19c-54105f8ae8a7>", "dump": "CC-MAIN-2013-20", "url": "http://daytimeroyaltyonline.com/single/?p=8906650&t=8780053", "date": "2013-05-18T05:48:59Z", "file_path": "s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz", "language": "en", "language_score": 0.8232095837593079, "token_count": 142}
            - save_path: Path to save the generated dataset
    """
    taxonomy_base = generation_config["taxonomy_base"]
    dataset_type = generation_config["dataset_type"]
    num_instances = generation_config["num_instances"]
    seed = generation_config["seed"]
    generator_llm_client: BaseLLMClient = generation_config["generator_llm_client"]
    document_sampling_pool_path = generation_config["document_sampling_pool_path"]
    save_path = generation_config.get(
        "save_path",
        f"user_data/{taxonomy_base}_{dataset_type}_{num_instances}_document_based.csv",
    )

    # generation-prompt selection
    if taxonomy_base == "datamorgana":
        if dataset_type == "qa":
            prompt = DM_QA_GENERATION_PROMPT_DOCUMENT_BASED
        elif dataset_type == "q":
            prompt = DM_QUESTION_GENERATION_PROMPT_DOCUMENT_BASED
        else:
            raise ValueError(
                f"Invalid dataset type: {dataset_type}; DataMorgana only supports 'qa' and 'q'."
            )
    elif taxonomy_base == "tuna":
        raise NotImplementedError(
            "TUNA-based dataset generation is not implemented yet."
        )
    else:
        raise ValueError(f"Invalid taxonomy base: {taxonomy_base}")

    rng = random.Random(seed)

    # Load documents from the sampling pool (jsonl format)
    with open(document_sampling_pool_path, "r", encoding="utf-8") as f:
        documents = [json.loads(line) for line in f if line.strip()]

    if not documents:
        raise ValueError(f"No documents found in pool: {document_sampling_pool_path}")

    # Pre-sample all documents and taxonomy types deterministically before any async work
    sampled_docs = [rng.choice(documents) for _ in range(num_instances)]

    if taxonomy_base == "datamorgana":
        sampled_q_types_list = [
            {
                cat: _sample_by_probability(types, rng)
                for cat, types in DM_QUESTION_TYPES.items()
            }
            for _ in range(num_instances)
        ]
        sampled_u_types_list = [
            {
                cat: _sample_by_probability(types, rng)
                for cat, types in DM_USER_TYPES.items()
            }
            for _ in range(num_instances)
        ]

    async def _generate_all() -> list:
        dataset = []
        for i in range(num_instances):
            doc = sampled_docs[i]
            document_text = doc.get("text", "")
            document_id = doc.get("id", str(i))

            if taxonomy_base == "datamorgana":
                sampled_q_types = sampled_q_types_list[i]
                sampled_u_types = sampled_u_types_list[i]

                # Build prompt keyword arguments from sampled types
                prompt_kwargs = {"document": document_text}
                for cat, t in sampled_q_types.items():
                    key = f"question_type_description_{cat.replace('-', '_')}"
                    prompt_kwargs[key] = t["description"]
                for cat, t in sampled_u_types.items():
                    key = f"user_type_description_{cat.replace('-', '_')}"
                    prompt_kwargs[key] = t["description"]

                filled_prompt = prompt.format(**prompt_kwargs)

                metadata = {
                    "document_id": document_id,
                    "user_types": {
                        cat: t["name"] for cat, t in sampled_u_types.items()
                    },
                    "question_types": {
                        cat: t["name"] for cat, t in sampled_q_types.items()
                    },
                }

            response = await generator_llm_client.generate(filled_prompt)

            if dataset_type == "qa":
                try:
                    parsed = json.loads(response)
                    question = parsed.get("question", "").strip()
                    answer = parsed.get("answer", "").strip()
                except (json.JSONDecodeError, AttributeError):
                    question = response.strip()
                    answer = ""
            else:  # dataset_type == "q"
                question = response.strip()
                answer = ""

            dataset.append(
                {
                    "qid": i + 1,
                    "metadata": metadata,
                    "problem": question,
                    "answer": answer,
                }
            )

        return dataset

    dataset = asyncio.run(_generate_all())

    # Save to CSV
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "metadata", "problem", "answer"])
        writer.writeheader()
        writer.writerows(dataset)

    return dataset


if __name__ == "__main__":
    # Example usage:
    generation_config = {
        "taxonomy_base": "datamorgana",
        "dataset_type": "qa",
        "num_instances": 10,
        "seed": 42,
        "generator_llm_client": BaseLLMClient(model="gpt-4o-mini"),
        "document_sampling_pool_path": "user_data/fineweb_500.jsonl",
    }
    dataset = generate_dataset_document_based(generation_config)
    print(dataset[0])
