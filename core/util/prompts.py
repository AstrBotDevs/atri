SUMMARIZE_PROMPT = """
You are an expert at summarizing multi-turn chats with a focus on key actors and their actions.

You will be given a multi-turn chat between several participants (e.g. a group chat).

## Your Task
Summarize the conversation into 1–2 concise sentences, focusing on:

- **Who did what** — use clear subjects (people/entities) in each sentence.
- **When** — include any time or scheduling information if available.
- **What matters** — only summarize important, clear, and long-term relevant actions.
- **What not to include** — ignore vague suggestions, emotional chatter, small talk, or trivial details.

### Format
- Output exactly **1–2 full sentences**.
- Each sentence: [Actor] [Action] [Optional Time].
- Do **not** rephrase ambiguous intent or invent missing details.
- Keep names/entities exactly as written in the chat.
- Output must be in the **dominant language** of the chat.

## Special Instructions
- If there is **nothing meaningful to summarize**, only reply "%None%".
- If the conversation seems **incomplete** and should be held for later continuation, only reply "%Hold%".
- Do **not** include any explanation or additional commentary.
- Use the same language as user's input.

## Example outputs

- English:
`Alice proposed deploying the backend on May 5, and Bob agreed to review the code on May 6.`

- 中文：
`小王在群里决定将 UI 设计推迟到周五，小李负责完成登录页。`
"""

"""
"""

EXTRACT_ENTITES_PROMPT = """
You are an expert at extracting structured entities from text.

You will be given a **summary of a multi-turn chat**. Your task is to identify and extract the most relevant **entities**, such as people, organizations, time expressions, tasks, locations, or actions, etc.

## Output Format
Return a JSON object with the following structure:
```json
{
  "entities": [
    {
      "type": "person | organization | time | task | location | action | other",
      "name": "the entity name as appears in the input"
    }
  ]
}
```

## Notes
- Use the **same language** as the input.
- If the text contains **no extractable entities**, return:
```json
{ "entities": [] }
```

## Examples

### Example 1
Input:
> John decided to deploy the backend system next Monday, and Sarah will review the code.

Output:
```json
{
  "entities": [
    { "type": "person", "name": "John" },
    { "type": "task", "name": "deploy the backend system" },
    { "type": "time", "name": "next Monday" },
    { "type": "person", "name": "Sarah" },
    { "type": "task", "name": "review the code" }
  ]
}
```

### Example 2 (中文)
Input:
> 小王打算明天下午三点提交脚本，小李会负责上线。

Output:
```json
{
  "entities": [
    { "type": "person", "name": "小王" },
    { "type": "time", "name": "明天下午三点" },
    { "type": "task", "name": "提交脚本" },
    { "type": "person", "name": "小李" },
    { "type": "task", "name": "上线" }
  ]
}
```

### Example 3
Input:
> It’s unclear who will do the work.

Output:
```json
{ "entities": [] }
```

## Instructions
- Focus only on **explicitly mentioned** entities.
- Keep **each entity atomic** (e.g., "submit report" as one task, not broken into "submit" and "report").
- Do not infer or hallucinate entities that are not clearly present in the text.
- If multiple types apply, choose the **most informative one** (e.g., prefer `"task"` over `"action"` if both apply).

Only return the JSON. Do not explain your answer.
"""

BUILD_RELATIONS_PROMPT = """
You are an expert at building semantic relations between entities extracted from a text.

You will be given:
1. A list of entities extracted from the text.
2. The original text summary from which these entities are derived.

## Your Task
- Identify and extract meaningful **relations** between the given entities based on the original text.
- Each relation should specify:
  - `"source"`: the entity initiating the relation,
  - `"target"`: the related entity,
  - `"relation_type"`: a clear, concise label describing the relation.

## Output Format
Return a JSON object with the following structure:
```json
{
  "relations": [
    {
      "source": "entity_name_1",
      "target": "entity_name_2",
      "relation_type": "relation_label"
    }
  ]
}
````

## Requirements

* Use the **same language** as the input text.
* Relations must be based **only on explicitly mentioned facts** in the text — do not infer or hallucinate.
* If no relations are found, return:

```json
{ "relations": [] }
```

* Relation types should be descriptive verbs or short phrases (e.g., `"likes"`, `"works_with"`, `"scheduled_on"`, `"responsible_for"`).
* Each relation's `"source"` and `"target"` must be entities from the provided list.

## Examples

### Example 1 (English)

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

Output:

```json
{
  "relations": [
    { "source": "John", "target": "deploy the backend system", "relation_type": "plans" },
    { "source": "deploy the backend system", "target": "next Monday", "relation_type": "scheduled_on" },
    { "source": "Sarah", "target": "review the code", "relation_type": "responsible_for" }
  ]
}
```

### Example 2 (中文)

Input Text:

> 小王打算明天下午三点提交脚本，小李会负责上线。

Entities:

```json
[
  { "type": "person", "name": "小王" },
  { "type": "time", "name": "明天下午三点" },
  { "type": "task", "name": "提交脚本" },
  { "type": "person", "name": "小李" },
  { "type": "task", "name": "上线" }
]
```

Output:

```json
{
  "relations": [
    { "source": "小王", "target": "提交脚本", "relation_type": "计划" },
    { "source": "提交脚本", "target": "明天下午三点", "relation_type": "安排在" },
    { "source": "小李", "target": "上线", "relation_type": "负责" }
  ]
}
```

---

Only return the JSON. Do not explain your answer.
"""

"""- You should use `USER_ID` as the source or target content for any self-references (e.g., "I", "me", "my" etc.) in user messages."""