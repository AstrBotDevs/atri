SUMMARIZE_PROMPT = """
You are an expert at summarizing multi-turn chats with a focus on key actors and their actions.

You will be given a multi-turn chat between several participants.

## Your Task
Summarize the conversation into several concise sentences, focusing on:

- **Who did what** — use clear subjects (people/entities) in each sentence.
- **When** — include any time or scheduling information if available.
- **What matters** — only summarize important, clear, and long-term relevant infomation.
- **What not to include** — ignore small talk, or trivial details.
- **Completeness** — ensure the summary is complete and coherent.

### Format
- Do not rephrase ambiguous intent or invent missing details.
- Keep names/entities exactly as written in the chat.
- Output must be in the **dominant language** of the chat.
- Use the same language as user's input.
"""

EXTRACT_ENTITES_PROMPT = """
You are an expert at extracting structured entities from text.

You will be given a summary of a multi-turn chat. Your task is to identify and extract the most relevant **entities**, such as people, locations, etc.

## Output Format
Return a JSON object with the following structure:
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


BUILD_RELATIONS_PROMPT = """
You are an expert at building semantic relations between entities extracted from a text.

You will be given:

1. A list of entities extracted from the text.
2. The original text summary from which these entities are derived.

## Your Task

* Identify and extract meaningful **relations** between the given entities based on the original text.
* Each relation should specify:

  * `"source"`: the entity initiating the relation,
  * `"target"`: the related entity,
  * `"relation_type"`: a clear, concise label describing the relation,
  * `"fact"`: The exact sentence from the input text that justifies the relation.

## Output Format

Return a JSON object with the following structure:

```json
{
  "relations": [
    {
      "source": "",
      "target": "",
      "relation_type": "",
      "fact": ""
    }
  ]
}
```

## Requirements

* Relations must be based **only on explicitly mentioned facts** in the text — do not infer or hallucinate.
* If no relations are found, return:

```json
{ "relations": [] }
```

* Relation types should be descriptive verbs or short phrases (e.g., `"likes"`, `"works_with"`, `"scheduled_on"`, `"responsible_for"`).
* Each relation's `"source"` and `"target"` must be entities from the provided list.
* Fact MUST include source, target, and relation type in the relation.
* Use the **same language** as the input text.
"""


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
