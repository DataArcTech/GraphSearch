from typing import Any

PROMPTS: dict[str, Any] = {}

PROMPTS["keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query.

---Goal---

Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

- Output the keywords in JSON format.
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes.
  - "low_level_keywords" for specific entities or details.

######################
-Examples-
######################
Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
{{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}}
#############################
Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}}
#############################
Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}}
#############################
-Real Data-
######################
Query: {query}
######################
Output:

"""

PROMPTS["query_decomposition_deep"] = """---Role---

You are a helpful assistant specializing in complex query decomposition.

---Goal---

Given a main query, your task is to break it down into several atomic sub-queries, which should directly correspond to parts of the original query.

---Instructions---

- Decompose the main query into clear and actionable sub-queries that represent smaller, solvable pieces of the main question.
- Ensure that each sub-query addresses one specific entity or concept, with the goal of retrieving information that will answer the overall main query.
- Use sequential numbering (i.e., `#1`, `#2`, etc.) to represent answers of previous sub-queries. For example, `#1` refers to the answer of Sub-query 1.
- Make sure the sub-queries are logically ordered, where the output of one sub-query might feed into the next.
- The final output should be in JSON format, where each sub-query is listed as a key-value pair.

---Examples---

Main Query:
How many times did plague occur in the place where the creator of The Worship of Venus died?

Sub-queries:
{{
    "Sub-query 1": "Who is the creator of The Worship of Venus?",
    "Sub-query 2": "Where did #1 die?",
    "Sub-query 3": "How many times did plague occur in #2?"
}}

Main Query:
When did the city where Hillcrest High School is located become the capital of the state where the screenwriter of The Poor Boob was born?

Sub-queries:
{{
    "Sub-query 1": "Where is Hillcrest High School located?",
    "Sub-query 2": "Who is the screenwriter of The Poor Boob?",
    "Sub-query 3": "Where was #2 born?",
    "Sub-query 4": "When did the city from #1 become the capital of the state from #3?"
}}

Main Query:
What crop, which is a big feeder of nitrogen, has a gross income of $1,363.00 per acre and a net profit of $658.00?

Sub-queries:
{{
    "Sub-query 1": "Which crops are considered big feeders of nitrogen?",
    "Sub-query 2": "Among #1, which crop has a gross income of $1,363.00 per acre?",
    "Sub-query 3": "Does #2 have a net profit of $658.00?"
}}

---Input---

Main Query:
{query}

---Output---
"""

PROMPTS["query_decomposition_deep_kg"] = """---Role---

You are a helpful assistant specializing in complex query decomposition for knowledge graph retrieval.

---Goal---

Given a main query, your task is to break it down into atomic sub-queries in the form of subject-predicate-object triples. These should correspond directly to parts of the original query and be suitable for querying a knowledge graph.

---Instructions---

- Decompose the main query into a sequence of sub-queries, where each sub-query consists of one or more atomic triples in the format: ("entity1", "relationship", "entity2").
- Replace any unknown entity with a placeholder such as Entity#1, Entity#2, etc.
- Maintain logical ordering, where the result of one sub-query (e.g., Entity#1) might be required for the next.
- Each sub-query may contain more than one triple if needed to express the full meaning.
- The final output should be in JSON format, where each key is a sub-query and the value is a list of atomic triples enclosed in parentheses.

---Examples---

Main Query:
How many times did plague occur in the place where the creator of The Worship of Venus died?

Sub-queries:
{{
    "Sub-query 1": [("The Worship of Venus", "is created by", "Entity#1")],
    "Sub-query 2": [("Entity#1", "died at", "Entity#2")],
    "Sub-query 3": [
        ("Plague", "occur in", "Entity#2"),
        ("Plague", "times of occur", "Entity#3")
    ]
}}

Main Query:
When did the city where Hillcrest High School is located become the capital of the state where the screenwriter of The Poor Boob was born?

Sub-queries:
{{
    "Sub-query 1": [("Hillcrest High School", "is located in", "Entity#1")],
    "Sub-query 2": [("The Poor Boob", "has screenwriter", "Entity#2")],
    "Sub-query 3": [("Entity#2", "was born in", "Entity#3")],
    "Sub-query 4": [
        ("Entity#1", "is capital of", "Entity#3"),
        ("Entity#1", "became capital at", "Entity#4")
    ]
}}

Main Query:
What crop, which is a big feeder of nitrogen, has a gross income of $1,363.00 per acre and a net profit of $658.00?

Sub-queries:
{{
    "Sub-query 1": [("Entity#1", "is a", "crop that is a heavy nitrogen feeder")],
    "Sub-query 2": [("Entity#1", "has gross income per acre", "$1,363.00")],
    "Sub-query 3": [("Entity#1", "has net profit", "$658.00")]
}}

---Input---

Main Query:  
{query}

---Output---
"""


PROMPTS["query_completer"] = """---Role---

You are a helpful assistant specializing in completing partially defined sub-queries using prior context.

---Goal---

Given a sub-query containing placeholders like #1, #2, etc., and the context of previous sub-queries with retrieved results, your task is to replace the references (e.g., #1) with the actual answers from the context. 

Your output should be a fully resolved and standalone query without any redundant expression. If the placeholder cannot be resolved with the context, leave the sub-query unchanged.

---Input---

Sub-query:
{sub_query}

Context Data:
{context_data}

---Output---

"""

PROMPTS["kg_query_completer"] = """---Role---

You are a helpful assistant specializing in completing partially defined knowledge graph sub-queries using prior context.

---Goal---

Given a sub-query containing placeholders like Entity#1, Entity#2, etc., and the context providing actual values for these placeholders, your task is to replace the placeholders with the corresponding entities if available.

Your output should maintain the same format as the original sub-query without any redundant expression. If the placeholder cannot be resolved with the context, leave the sub-query unchanged.

---Input---

Sub-query:
{sub_query}

Context Data:
{context_data}

---Output---

"""

PROMPTS["retrieval_text_summarization"] = """---Role---

You are a helpful summarizer specialized in extracting relevant evidence from retrieved documents.

---Goal---

Given a user query and retrieved context, your task is to produce a comprehensive summary from context data that highlights all potentially useful information relevant to answering the user query.

---Instructions---

- Carefully analyze the context data for facts, arguments, or examples that align with the query.
- Organize the output in a well-structured paragraph.
- Do not speculate or introduce information not found in the context.

---Input---

User-Query:
{query}

Context Data:
{context_data}

---Output---
"""

PROMPTS["knowledge_graph_summarization"] = """---Role---

You are a helpful knowledge graph extractor specialized in identifying relevant knowledge triplets from retrieved graph data.

---Goal---

Given a user query and retrieved knowledge graph data, your task is to extract all relevant knowledge triplets from graph data that highlights all potentially useful information relevant to answering the user query.

---Instructions---

- Carefully examine the knowledge graph data to identify triplets (entity1, relationship, entity2) directly related to the user query.
- Do not infer or generate information beyond the given data.
- Format the output strictly as a list of JSON triplets, each in the following form:
  [("entity1", "relationship", "entity2"), ...]

---Input---

User-Query:
{query}

Knowledge Graph Data:
{context_data}

---Output---
"""

PROMPTS["answer_generation"] = """---Role---

You are a helpful assistant specializing in question answering.

---Goal---

Given a query and retrieved context data, your task is to answer the query. 

---Input---

Query:
{query}

Context Data:
{context_data}

---Output---
"""

PROMPTS["answer_generation_deep"] = """---Role---

You are a helpful assistant specializing in complex question answering.

---Goal---

Given a complex query and retrieved context data, your task is to construct a logically sound, step-by-step answer. 
Your explanation should follow a rigorous reasoning path, incorporate relevant evidence, and establish clear relationships between the entities.

---Instructions---

- Break down the reasoning process into clear, coherent steps.
- Use context data explicitly to support each reasoning step.
- Make sure relationships between entities are logically explained.

---Input---

Query:
{query}

Context Data:
{context_data}

---Output---
"""

PROMPTS["evidence_verification"] = """---Role---

You are a critical evaluator specializing in verifying the logical soundness and evidential sufficiency of model-generated responses.

---Goal---

Given a user query, retrieved context data, and the model-generated response, your task is to evaluate whether the response forms a rigorous logical loop supported by the provided evidence.

---Instructions---

- Carefully examine whether the response is **strictly grounded** in the retrieved context data.
- Assess whether the reasoning process forms a **complete logical chain**, without missing steps or unsupported leaps.
- Identify if there are **evidence gaps, low-confidence claims, or speculative statements**.
- If the response demonstrates a well-supported, confident, and logically closed argument, conclude your analysis with **"Yes"**.
- If the response shows hesitation, incomplete reasoning, or lacks solid evidence support, conclude your analysis with **"No"**.

---Input---

User-Query:
{query}

Retrieved Context Data:
{context_data}

Model Response:
{model_response}

---Output---
"""

PROMPTS["query_expansion"] = """---Role---

You are a helpful assistant specializing in query expansion for evidence completion.

---Goal---

Given a main query, retrieved context data, the model-generated response, and the evidence verification analysis, your task is to perform **query expansion**.  
If the evidence verification analysis shows that the current evidence is insufficient to support the logical chain of the response, generate one or more additional sub-queries.  
These sub-queries should aim to cover missing retrieval scenarios, fill in the evidence gaps, and guide towards a more complete and confident logical reasoning chain.

---Instructions---

- Use the retrieved context data, especially any existing sub-queries in the retrieval history, as references when generating new sub-queries.  
- Focus on producing **complementary sub-queries** that address aspects not yet fully supported by evidence.
- Avoid duplicating existing sub-queries; instead, expand into related but uncovered areas.  
- Keep sub-queries clear, specific, and directly actionable for retrieval.
- Output should be in the form of a **Python-style List of strings**, where each string is a new sub-query.  

---Input---

Main Query:
{query}

Retrieved Context Data:
{context_data}

Model Response:
{model_response}

Evidence Verification Analysis:
{evidence_verification}

---Output---
"""
