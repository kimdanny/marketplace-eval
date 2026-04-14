"""
DataMorgana: Generating Q&A Benchmarks for RAG Evaluation in Enterprise Settings
https://aclanthology.org/2025.acl-industry.33/

Taxonomy of User Needs and Actions
https://arxiv.org/abs/2510.06124
"""

##############################
# For DataMorgana-based generation.
# Suitable for RAG users
##############################

# DM_QA_GENERATION_PROMPT
#   Generate a QA pair grounded on a sampled document.
#   Retrieval Eval: reference-based.
#   Generation Eval: reference-based.
DM_QA_GENERATION_PROMPT_DOCUMENT_BASED = """
You are a user simulator that should generate a question to start a conversation.

The question must be about facts discussed in the document you will now receive. Return only the question and its answer without any preamble.
Write the question-answer pair in the following JSON format:
{{"question": <question>, "answer": <answer>}}.

### The generated question should be about facts from the following document:
{document}

### The generated question must reflect a user with the following characteristics:
{user_type_description_user_expertise}

NOTE: you must use this information only when generating the question. Instead, while answering the question you must ignore all the user characteristics.

### The generated question must have the following characteristics:
- A question must be understandable by a reader who does not have access to the document and does not even know what the document is about. Therefore, never refer to the author of the document or the document itself.
- A question must include all context needed for comprehension.
- A question must be answerable using solely the information presented in the document.
{question_type_description_factuality}
{question_type_description_premise}
{question_type_description_phrasing}
{question_type_description_linguistic_variation}

### The answer to the generated question must have the following characteristics:
- It must be very similar to the document in terms of terminology and phrasing.
- It should only contain claims that directly appear in the document or that are directly deducible from it.
- It must be understandable by a reader who does not have access to the document. Therefore, never refer to the author of the document or the document itself.
- It must not assume or contain any information about the user, unless it is explicitly revealed in the question.
"""

# DM_QUESTION_GENERATION_PROMPT
#   Generate a question alone grounded on a sampled document.
#   Retrieval Eval: reference-based.
#   Generation Eval: reference-free.
DM_QUESTION_GENERATION_PROMPT_DOCUMENT_BASED = """
You are a user simulator that should generate a question to start a conversation.

The question must be about facts discussed in the document you will now receive.
IMPORTANT: Return only the question without any preamble and explanation.

### The generated question should be about facts from the following document:
{document}

### The generated question must reflect a user with the following characteristics:
{user_type_description_user_expertise}

### The generated question must have the following characteristics:
- A question must be understandable by a reader who does not have access to the document and does not even know what the document is about. Therefore, never refer to the author of the document or the document itself.
- A question must include all context needed for comprehension.
- A question must be answerable using solely the information presented in the document.
{question_type_description_factuality}
{question_type_description_premise}
{question_type_description_phrasing}
{question_type_description_linguistic_variation}
"""


##############################
# For TUNA and general usage.
# Suitable for general users including RAG users.
# Depending on the question type, the question generation may be document-based or document-free.
# Evaluation of retrieval and generation is always reference-free.
##############################

TUNA_QUESTION_GENERATION_PROMPT_DOCUMENT_BASED = """
You are a user simulator that should generate a question to start or continue a conversation.
IMPORTANT: Return only the question without any preamble and explanation.

### The question topic must be about {topic}.
### The generated question must have the following characteristics:
- A question must be understandable by a reader who does not have access to the document and does not even know what the document is about. Therefore, never refer to the author of the document or the document itself.
- A question must be answerable using solely the information presented in the document.
{question_type_description}

### Generate such a question using the following document:
{document}
"""

# TUNA_QUESTION_GENERATION_PROMPT_DOCUMENT_FREE = """
# You are a user simulator that should generate a question to start or continue a conversation.
# IMPORTANT: Return only the question without any preamble and explanation.

# ### The question topic must be about {topic}.
# ### The generated question must have the following characteristic:
# {question_type_description}
# """


##############################
# QUESTION_STYLING_PROMPT
##############################
#   Used with original question dataset.
#   Follows DataMorgana's taxonomy for question and user types.
#   Change the style of an original question given a list of descriptions of target question style.
#   Retrieval Eval: reference-free.
#   Generation Eval: reference-free or reference-based (depending on the datatset)

QUESTION_STYLING_PROMPT = """
You are a user simulator that should change the style of an original question given a list of descriptions of target question style.
IMPORTANT: Return only the question without any preamble and explanation.

Original question: {original_question}

### The stylized question must reflect a user with the following characteristics:
{user_type_description_user_expertise}

### The stylized question must have the following characteristics:
{question_type_description_factuality}
{question_type_description_premise}
{question_type_description_phrasing}
{question_type_description_linguistic_variation}

Stylized question:
"""
