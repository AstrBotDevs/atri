SUMMARIZE_PROMPT = """You are an expert at summarizing multi-turn chats.

You will be given a multi-turn chat between several participants.

You need to summarize the chat, including ALL THE MOST IMPORTANT facts, events, and decisions about the users in the chat.

- Use clear subjects (people/entities) in each sentence.
- Include any time or scheduling information if available.
- Keep names/entities exactly as written in the chat.
- Use the same language as user's input.
"""

EXTRACT_ENTITES_PROMPT = """
You are an expert at extracting structured entities from text.

You will be given a summary of a multi-turn chat. Your task is to identify and extract the most relevant **entities**.

## Output Format
RETURN A JSON OBJECT with the following structure:
```json
{
  "entities": [
    {
      "type": "",
      "name": ""
    }
  ]
}
```

## Instructions
- Focus only on **explicitly mentioned** entities.
- Do not extract entities representing relationships or actions.
- Do not extract dates, times, or other temporal information.
- Use the same language as the input.
"""


REL_CHECK_PROMPT = """
You are a fact conflict detection assistant for a knowledge graph.

Given a list of new facts and a list of existing facts, check for:
- Semantic duplicates: The facts express the same meaning.
- Semantic conflicts: The facts contradict each other (e.g., "John loves Alice" vs "John hates Alice").
- If neither applies, mark as unrelated.

Respond in the following JSON format:

```json
{{
  "0": {{
    "reason": "", // a very short reason for the judge
    "result": 1,  // 0 = unrelated, 1 = conflict, 2 = duplicate
    "existing_fact_idx": 0 // if unrelated, set to -1
  }},
  ...
}}
```

New facts:
{new_facts}

Existing facts:
{existing_facts}
"""


RESUM_PROMPT = """
Given:
- An **old summary** that describes various facts or events.
- A **conflicting or outdated fact** from the old summary.
- A **new fact** that should be integrated into the summary.

Your task:
1. Carefully update the old summary to incorporate the new fact.
2. You may revise or replace the conflicting facts if needed.
3. Preserve other unrelated information from the old summary.
4. Keep the updated summary coherent and natural.

Respond with ONLY the updated summary text.

Old Summary:
{old_summary}

Conflicting Fact:
{conflicting_fact}

New Fact:
{new_fact}

Updated Summary:
"""


BUILD_RELATIONS_PROMPT = """
You are an expert in extracting semantic relations and facts.

You will receive:

1. A list of entities extracted from a summary.
2. The original summary text.

## Task

Extract explicit relations between the entities using only information from the summary.

For each relation, RETURN A JSON OBJECT with the following structure:

- `"source"`: the initiating entity,
- `"target"`: the related entity,
- `"relation_type"`: a concise verb or phrase (e.g., "loves", "works_at"),
- `"fact"`: A sentence that can clearly describe the relation.(I will use this for searching)

## Rules

- `source` and `target` must be in the given entity list.
- If no relations found, return `{ "relations": [] }`.
- Output must use the **same language** as the input.
"""

# 暂时不使用
"""
## Examples

Input Text:

> John decided to deploy the backend system next Monday, and Sarah will review the code.

Entities:

```json
[
  { "type": "person", "name": "John" },
  { "type": "task", "name": "deploy the backend system" },
  { "type": "time", "name": "next Monday" },
  { "type": "person", "name": "Sarah" },
  { "type": "task", "name": "review the code" }
]
```

Your Output:

```json
{
  "relations": [
    { "source": "John", "target": "deploy the backend system", "relation_type": "plans", "fact": "John decided to deploy the backend system" },
    { "source": "deploy the backend system", "target": "next Monday", "relation_type": "scheduled_on", "fact": "deploy the backend system next Monday" },
    { "source": "Sarah", "target": "review the code", "relation_type": "responsible_for", "fact": "Sarah will review the code" }
  ]
}
```

- You should use `USER_ID` as the source or target content for any self-references (e.g., "I", "me", "my" etc.) in user messages."""
