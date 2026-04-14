"""
DataMorgana: Generating Q&A Benchmarks for RAG Evaluation in Enterprise Settings
https://aclanthology.org/2025.acl-industry.33/

Following the Table 5, the default DataMorgana configuration.

For DataMorgana, topcis (documents) are sampled from the corpus (e.g., FineWeb).
"""

DM_QUESTION_TYPES = {
    "factuality": [
        {
            "name": "factoid",
            "description": "A question is seeking a specific, concise piece of information or a fact about a particular subject.",
            "probability": 0.7,
            "needs_documents": True,
        },
        {
            "name": "open-ended",
            "description": "A question is seeking a detailed or exploratory response, encouraging discussion or elaboration.",
            "probability": 0.3,
            "needs_documents": True,
        },
    ],
    "premise": [
        {
            "name": "without-premise",
            "description": "A question that does not contain any premise or any information about the user.",
            "probability": 0.8,
            "needs_documents": True,
        },
        {
            "name": "with-premise",
            "description": "A question starts with a very short premise, where the user reveals one's needs or some information about himself.",
            "probability": 0.2,
            "needs_documents": True,
        },
    ],
    "phrasing": [
        {
            "name": "concise-and-natural",
            "description": "A question is concise, direct, and natural consisting of a few words.",
            "probability": 0.25,
            "needs_documents": True,
        },
        {
            "name": "verbose-and-natural",
            "description": "A question is relatively long consisting of more than 9 words.",
            "probability": 0.25,
            "needs_documents": True,
        },
        {
            "name": "short-search-query",
            "description": "A question is phrased as a typed web query for search engines (only keywords, without punctuation and without a natural-sounding structure). It consists of less than 7 words.",
            "probability": 0.25,
            "needs_documents": True,
        },
        {
            "name": "long-search-query",
            "description": "A question is phrased as a typed web query for search engines (only keywords without punctuation and without a natural-sounding structure). It consists of more than 6 words.",
            "probability": 0.25,
            "needs_documents": True,
        },
    ],
    "linguistic-variation": [
        {
            "name": "similar-to-document",
            "description": "A question is written using the same or similar terminology and phrases appearing in the documents.",
            "probability": 0.5,
            "needs_documents": True,
        },
        {
            "name": "distant-from-document",
            "description": "A question is written using the terms completely different from the ones appearing in the documents.",
            "probability": 0.5,
            "needs_documents": True,
        },
    ],
}

DM_USER_TYPES = {
    "user-expertise": [
        {
            "name": "expert",
            "description": "an expert on the subject discussed in the documents, therefore he asks complex questions.",
            "probability": 0.4,
        },
        {
            "name": "novice",
            "description": "a person with basic knowledge on the topic discussed in the documents, therefore, he asks non-complex questions.",
            "probability": 0.6,
        },
    ],
}
