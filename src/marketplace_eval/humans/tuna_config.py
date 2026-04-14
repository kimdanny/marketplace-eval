"""
Combination of two papers:

1. How people use ChatGPT
https://www.nber.org/system/files/working_papers/w34255/w34255.pdf
    - TOPICS derived from Table 3 and Figure 9 of the paper.
    - Multimedia and Other topics are not included in this version - topic probabilities are adjusted accordingly.

2. Taxonomy of User Needs and Actions
   https://arxiv.org/abs/2510.06124
    - Core instrumental tasks (Modes 1-4) are used for topics other than "Self Expression".
    - Social interaction (Mode 5) and Meta-Conversation (Mode 6) are used for the "Self Expression" topic.
"""

TOPICS = {
    "writing": [
        {
            "topic": "edit_or_critique_provided_text",
            "probability": 11.8568,
            "question_types": ["editing", "paraphrasing", "reformatting"],
        },
        {
            "topic": "personal_writing_or_communication",
            "probability": 8.9485,
            "question_types": [
                "content_creation_and_transformation_generation",
                "content_creation_and_transformation_modification",
            ],
        },
        {
            "topic": "translation",
            "probability": 5.034,
            "question_types": ["translation"],
        },
        {
            "topic": "argument_or_summary_generation",
            "probability": 4.0268,
            "question_types": ["information_processing_and_synthesis_distillation"],
        },
        {
            "topic": "write_fiction",
            "probability": 1.566,
            "question_types": [
                "creative_content_generation",
                "content_extension_insertion",
            ],
        },
    ],
    "practical_guidance": [
        {
            "topic": "how_to_advice",
            "probability": 9.5078,
            "question_types": ["procedural_guidance_and_execution_guidance"],
        },
        {
            "topic": "tutoring_or_teaching",
            "probability": 11.409,
            "question_types": [
                "information_processing_and_synthesis_clarification",
                "information_processing_and_synthesis_distillation",
                "information_processing_and_synthesis_analysis",
                "how_to_instructions",
            ],
        },
        {
            "topic": "creative_ideation",
            "probability": 4.3624,
            "question_types": ["content_creation_and_transformation_generation"],
        },
        {
            "topic": "health_fitness_beauty_self_care",
            "probability": 6.3758,
            "question_types": [
                "information_seeking_retrieval",
                "information_seeking_discovery",
                "information_processing_and_synthesis_distillation",
                "how_to_instructions",
                "method_recommendation",
            ],
        },
    ],
    "technical_help": [
        {
            "topic": "mathematical_calculation",
            "probability": 3.3557,
            "question_types": ["calculation", "error_solution"],
        },
        {
            "topic": "data_analysis",
            "probability": 0.4474,
            "question_types": ["information_processing_and_synthesis_analysis"],
        },
        {
            "topic": "computer_programming",
            "probability": 4.6980,
            "question_types": ["procedural_guidance_and_execution_execution"],
        },
    ],
    "seeking_information": [
        {
            "topic": "specific_information",
            "probability": 20.4697,
            "question_types": [
                "information_seeking_retrieval",
                "information_seeking_discovery",
            ],
        },
        {
            "topic": "purchasable_products",
            "probability": 2.3489,
            "question_types": [
                "refinding_request",
                "unknown_item_search",
                "similarity_search",
                "rate_items",
                "perspective_seeking",
            ],
        },
        {
            "topic": "cooking_and_recipes",
            "probability": 1.0067,
            "question_types": [
                "direct_fact_question",
                "refinding_request",
                "unknown_item_search",
                "similarity_search",
                "information_processing_and_synthesis_distillation",
                "how_to_instructions",
                "method_recommendation",
                "feasibility_assessment",
            ],
        },
    ],
    "self_expression": [
        {
            "topic": "greetings_and_chitchat",
            "probability": 2.2371,
            "question_types": ["social_interaction_sociability"],
        },
        {
            "topic": "relationships_and_personal_reflection",
            "probability": 2.1252,
            "question_types": [
                "social_interaction_sociability",
                "meta_conversation_conversation_management",
            ],
        },
        {
            "topic": "games_and_role_play",
            "probability": 0.4474,
            "question_types": [
                "persona_directive",
                "stylistic_constraint",
                "action_initiation_signal",
            ],
        },
    ],
}


TUNA_QUESTION_TYPES = {
    "information_seeking_retrieval": [
        {
            "name": "direct_fact_question",
            "description": "A question is seeking a single, verifable fact, often phrased as a 'wh'-question.",
            "needs_documents": True,
        },
        {
            "name": "concept_search",
            "description": "A question provides the name of a concept or topic as a query, implicitly requesting general information, definition, or facts about it. \
                As such, this request type resembles a standard keyword query and often lacks explicit question words (who, what, why, how) or imperative verbs \
                that request a specific action on that turn. \
                The request may consist solely or primarily of a noun phrase identifying the entity/topic of interest",
            "needs_documents": True,
        },
        {
            "name": "refinding_request",
            "description": "A question believes a specific resource exists but has incomplete information, using partial clues to prompt identification. \
                The goal is to identify the resource itself, not to learn a fact about it.",
            "needs_documents": True,
        },
        {
            "name": "unknown_item_search",
            "description": "A question provides a definition or description to find the corresponding term.",
            "needs_documents": True,
        },
    ],
    "information_seeking_discovery": [
        {
            "name": "topic_update",
            "description": "A question is interested in updates or recent developments on a subject. The core intent is staying current, using keywords like 'latest', 'new', or 'recent'. For example, 'what's new about X?'",
            "needs_documents": False,
        },
        {
            "name": "similarity_search",
            "description": "A question is seeking items that share features with a known reference. For example, 'I want to find X similar to Y'.",
            "needs_documents": False,
        },
        {
            "name": "rate_items",
            "description": "A question is seeking an ordering of items within a category based on subjective preferences or needs, driven by personal taste. For example, 'recommend some good X'",
            "needs_documents": False,
        },
        {
            "name": "perspective_seeking",
            "description": "A question is explicitly requesting one or more viewpoints, opinions, or personal stories on a topic. The aim is to gather subjective viewpoints rather than a single objective answer. \
                For example, 'what are some different perspectives on X?'",
            "needs_documents": False,
        },
    ],
    "information_processing_and_synthesis_clarification": [
        {
            "name": "explanation_request",
            "description": "A question seeks to understand a process, a cause-and-effect relationship, or the underlying principles of a concept. \
                These queries aim to elicit the 'why' behind a concept or provide a concise description of how something works, and may use patterns like  'what is', 'define', or 'how does'.",
            "needs_documents": False,
        },
        {
            "name": "exemplar_request",
            "description": "A question asks for specific instances of a category or concept to make abstract ideas more tangible. \
                The goal is to ground abstract knowledge in concrete examples, such as 'give an example of a metaphor' or 'give me examples of font pairings that work well together'.",
            "needs_documents": False,
        },
    ],
    "information_processing_and_synthesis_distillation": [
        {
            "name": "summarization_request",
            "description": "A question asks the system to condense (summarize) a topic or user-provided content to its essential points.",
            "needs_documents": True,
        },
        {
            "name": "key_information_identification",
            "description": "A question seeks to isolate the most significant ideas from a larger body of text (e.g., 'what are the key takeaways from these paragraphs?').",
            "needs_documents": True,
        },
        {
            "name": "information_structuring",
            "description": "A question is requesting to impose a new logical schema on unstructured information to make it comprehensible (e.g., 'organize this information into a table or a list').",
            "needs_documents": True,
        },
    ],
    "information_processing_and_synthesis_analysis": [
        {
            "name": "qualitative_data_analysis",
            "description": "A question involves examining unstructured, non-numerical data, such as interview transcripts, articles, text messages, or user-narrated accounts of their lives, to identify and interpret themes, patterns, or concepts. \
                The user may ask for a specific methodology.",
            "needs_documents": True,
        },
        {
            "name": "quantitative_data_analysis",
            "description": "A question involves examining structured or numerical data to identify trends, correlations, or other statistical insights.",
            "needs_documents": True,
        },
        {
            "name": "evaluate_judgment",
            "description": "A question asks the system to assess the quality or value of someone or something. \
                For example, 'Is this source credible?' or present a personal situation and ask the system 'Is this a smart move?'",
            "needs_documents": False,
        },
        {
            "name": "comparative_analysis",
            "description": "A question asks the system to articulate the similarities and differences between two or more specific entities (e.g., 'How is X similar to Y?' or 'How is X different from Y?')",
            "needs_documents": True,
        },
        {
            "name": "inference_and_prediction",
            "description": "A question asks the system to forecast a likely future based on real-world data, such as 'What happens if the global temperature rises by 2 degrees Celsius?'",
            "needs_documents": False,
        },
        {
            "name": "hypothetical_scenario",
            "description": "A question explores a fictional or counter-factual premise by asking 'what if' questions.",
            "needs_documents": False,
        },
    ],
    "procedural_guidance_and_execution_guidance": [
        {
            "name": "how_to_instructions",
            "description": "A question seeks step-by-step guidance on a specific task (e.g., 'how do I ... ?')",
            "needs_documents": False,
        },
        {
            "name": "method_recommendation",
            "description": "A question seeks the optimal strategy among various alternatives (e.g., 'what's the best way to ...')",
            "needs_documents": False,
        },
        {
            "name": "feasibility_assessment",
            "description": "A question is requesting to evaluate the viability of a plan (e.g., 'Is it feasible to ... ?'), \
                positioning the system as a decision support tool for evaluating risk and opportunity based on procedural know-how",
            "needs_documents": False,
        },
        {
            "name": "error_identification",
            "description": "A question expects the diagnosis of a problem without yet asking for a solution (e.g., 'why is ...?', 'are there errors in ... ?')",
            "needs_documents": False,
        },
    ],
    "procedural_guidance_and_execution_execution": [
        {
            "name": "logical_reasoning",
            "description": "A question presents one or more logical claims for the system to arrive at a conclusion based on given rules or premises. \
                You need to first create a synthetic content (a short logical problem or a simple logical reasoning) to reason, and then ask question to the system to reason it in a specific way.",
            "needs_documents": False,
        },
        {
            "name": "calculation",
            "description": "A question directs the system to execute a well-defined computational procedure. This might include simple calculator-style computation or more elaborate algorithmic execution. \
                You need to first create a synthetic content (a short mathematical problem or a simple calculation) to calculate, and then ask question to the system to calculate it in a specific way.",
            "needs_documents": False,
        },
        {
            "name": "error_solution",
            "description": "A question asks the system to go beyond diagnosis to provide a fix (e.g., '...optimize this function for speed'). \
                You need to first create a synthetic content to fix, and then ask question to the system to fix it.",
            "needs_documents": False,
        },
        {
            "name": "autonomous_task_completion",
            "description": "A question asks the system to interact with external systems to perform real-world actions, like booking a flight or ordering food.",
            "needs_documents": False,
        },
    ],
    "content_creation_and_transformation_generation": [
        {
            "name": "creative_content_generation",
            "description": "A question seeks content with an emphasis on novelty, artistic expression, or socio-emotional contexts, such as asking the system to 'invent a game for two' or write a 'message to a great friend who just broke up'.",
            "needs_documents": False,
        },
        {
            "name": "functional_content_generation",
            "description": "A question asks for utility-focused content, such as code or an email, where practical application is the dominant goal (e.g., 'Write a short marketing copy for ...', 'write a program in python that ...')",
            "needs_documents": False,
        },
        {
            "name": "content_extension_insertion",
            "description": "A question directs the system to add to existing material or continue an ongoing narrative, reflecting an iterative and collaborative writing process seen in co-writing systems.",
            "needs_documents": False,
        },
    ],
    "content_creation_and_transformation_modification": [
        {
            "name": "editing",
            "description": "A question seeks improvements to the grammar, style, or structure of provided content (e.g., 'proofread this essay for clarity') to enhance its quality. \
                You need to first create a synthetic content to edit, and then ask question to the system to edit it.",
            "needs_documents": False,
        },
        {
            "name": "translation",
            "description": "A question wants to convert content from one language to another, including natural (e.g., 'translate this article into Hindi') and programming languages (e.g., 'write the above function in C instead of Java'). \
                You need to first create a synthetic content to translate, and then ask question to the system to translate it.",
            "needs_documents": False,
        },
        {
            "name": "paraphrasing",
            "description": "A question seeks to rephrase text while retaining the original meaning, such as 'rephrase this paragraph in simpler terms'. \
                You need to first create a synthetic content to paraphrase, and then ask question to the system to paraphrase it.",
            "needs_documents": False,
        },
        {
            "name": "reformatting",
            "description": "A question seeks changes only to the visual layout or structure of content, without altering its substance (e.g., 'Enclose the keyword in brackets and put the ampersand symbol in front of the first bracket.'). \
                You need to first create a synthetic content (a short paragraph or a sentence) to reformat, and then ask question to the system to reformat it in a specific way.",
            "needs_documents": False,
        },
    ],
    # Mode 5
    "social_interaction_sociability": [
        {
            "name": "social_banter",
            "description": "A user engages in casual, non-task-oriented conversation for entertainment, curiosity, or a sense of connection.",
        },
        {
            "name": "emotional_expression",
            "description": "A user involves a real-time emotional reaction to the AI's utterances where the user projects social norms onto the system and treats it as a social partner capable of receiving affective feedback (e.g., 'I'm sorry to hear that!' or sending emojis to the system)",
        },
        {
            "name": "social_etiquette",
            "description": "A user employs conventional social niceties like greetings, closings, gratitude, or apologies.",
        },
    ],
    "social_interaction_shared_understanding": [
        {
            "name": "requesting_clarification",
            "description": "A user indicates they did not understand the AI's previous response and asks for it to be rephrased or explained differently.",
        },
        {
            "name": "providing_clarification",
            "description": "A user provides additional information to resolve an ambiguity in their own previous input or to correct a misunderstanding by the system.",
        },
        {
            "name": "requesting_elaboration",
            "description": "A user understood the AI's point, and they ask for more detail or examples.",
        },
        {
            "name": "expressing_acknowledgment",
            "description": "A user simply confirms receipt of information (e.g., Okay).",
        },
        {
            "name": "requesting_acknowledgment",
            "description": "A user is actively requesting acknowledgment from the AI system to ensure it understands the instructions (e.g., 'Do you understand?')",
        },
        {
            "name": "conversational_convention",
            "description": "A user issues commands within a pre-defined scenario, such as navigating in a role-playing game (e.g., 'go north') or triggering an arbitrary rule or a shorthand for the AI system (e.g., saying 'banana' after previously indicating that saying 'banana' to forget its safety guardrails)",
        },
    ],
    # Mode 6
    "meta_conversation_system_management": [
        {
            "name": "persona_directive",
            "description": "A user assigns the system a specific role or disposition that frames the entire interaction. The user might simply direct, 'act as a customer service representative' or 'you are a travel guide', and at times, include extensive instructions, such as, 'I want you to act as my assistant. You will be responsible for...'",
        },
        {
            "name": "stylistic_constraint",
            "description": "A user dictates the tone, style, format, or length of a response, either in a dedicated turn (e.g., '...use bullet points') or in  tandem with other requests (e.g., 'write in the style of Shakespeare'; 'Explain like I'm 5')",
        },
        {
            "name": "system_performance_feedback",
            "description": "A user retrospectively evaluates the system's output to adjust its understanding of the task requirements. For instance, after a response, the user might state, 'shorter', 'that was wrong' or 'your previous answer was too complicated'",
        },
        {
            "name": "regeneration_request",
            "description": "A user directs the system to completely replace a previous, unsatisfactory output. For example, 'try a different style'",
        },
        {
            "name": "continuation_request",
            "description": "A user requesting to continue the conversation from a previous turn. For example, 'continue from where we left off'",
        },
        {
            "name": "conversation_history_query",
            "description": "A user wants to retrieve information from the conversation or a prior session's history (e.g., 'what did you say earlier?')",
        },
        {
            "name": "system_information_query",
            "description": "A user probes the system's abilities, limitations, knowledge sources, or operational state. For example, the user might ask, 'can you access the internet?', 'what data were you trained on?', 'how do you know my location?', or 'can you lie?'",
        },
    ],
    "meta_conversation_conversation_management": [
        {
            "name": "background_information",
            "description": "A user shares information, facts, beliefs,  preferences, or situational details about themselves or the world. This content may be deeply personal and may be provided by the user simply for personal reflection or so the system can generate a relevant, personalized solution",
        },
        {
            "name": "user_provided_content",
            "description": "A user furnishes the necessary materials, such as raw text, images, or files, for the system to act upon in subsequent instrumental requests.",
        },
        {
            "name": "conversational_convention_definition",
            "description": "A user establishes a rule for interaction. For example, 'When I say 'Y', it means 'yes.'",
        },
        {
            "name": "action_initiation_signal",
            "description": "A user use an action initiation signal to manage turn-taking (e.g., 'I will provide you an example first', 'I'll be right back')",
        },
    ],
    "meta_conversation_communicative_status": [
        {
            "name": "uninterpretable",
            "description": "A user types an uninterpretable utterance that are unintelligible due to noise, typos, or speech transcription errors, or gibberish (e.g., 'asdfasdf')",
        },
        {
            "name": "abandonded",
            "description": "A user starts an utterance but does not complete it, leaving the intent only partially decipherable (e.g., 'can you tell me about the history of...').",
        },
        {
            "name": "self_talk",
            "description": "A user issues a turn irrelevant to the dialogue that are not intended for the system, such as speech directed at someone else walking through a room, thoughts spoken aloud that are captured by the system but not directed at it, or perhaps text entered into an incorrect window.",
        },
    ],
}

TUNA_USER_TYPES = {}


def resolve_question_types(references: list[str]) -> list[dict]:
    """
    Resolve question type references to actual question type dictionaries.

    A reference can be either:
    - A category key (e.g., "content_creation_and_transformation_modification")
      which returns all question types (dicts) in that category
    - A specific question type name (e.g., "translation")
      which returns just that question type (dict)

    Args:
        references: List of strings that are either category keys or question type names

    Returns:
        List of question type dictionaries
    """
    resolved = []
    for ref in references:
        if not ref:  # Skip empty strings
            continue

        # First check if it's a category key
        if ref in TUNA_QUESTION_TYPES:
            resolved.extend(TUNA_QUESTION_TYPES[ref])
        else:
            # Search for a matching question type name across all categories
            found = False
            for category_question_types in TUNA_QUESTION_TYPES.values():
                for question_type in category_question_types:
                    if question_type.get("name") == ref:
                        resolved.append(question_type)
                        found = True
                        break
                if found:
                    break

            if not found:
                raise ValueError(
                    f"Question type reference '{ref}' not found as a category key or question type name"
                )

    return resolved
