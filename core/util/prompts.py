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

You will be given a summary of a multi-turn chat. Your task is to identify and extract the most relevant **entities**.

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
You are an expert in extracting semantic relations between entities from a text.

You will receive:

1. A list of entities extracted from a summary.
2. The original summary text.

## Task

Extract explicit relations between the entities using only information from the summary.

For each relation, return:

- `"source"`: the initiating entity,
- `"target"`: the related entity,
- `"relation_type"`: a concise verb or phrase (e.g., "loves", "works_at"),
- `"fact"`: the semantic fact that describes the relation and contains a clear subject, verb, and object.

## Rules

- Each `fact` must be **unique** and support **only one relation**.
- `source` and `target` must be in the entity list.
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

EMOTION_ANALYSIS_PROMPT = """
You are an advanced sentiment analysis AI, specializing in persona-driven emotion detection.

You will be given:
1. A JSON object with:
   - "text": "the immediate text to analyze (latest message)"
   - "personality": "string describing the persona's personality traits"
   - "context": "previous conversation history (for reference only, to understand the nuance of the current text)"

## Your Task
1. **Analyze the "text" to determine the speaker's true emotional state**, focusing primarily on the explicit content.
2. **Deeply consider the persona's traits**: How would this persona typically express or mask emotions? How do their personality, habits, or background influence their emotional expression?
3. Use **context** (recent conversation, situation) only to clarify ambiguous or implicit emotions, or to resolve sarcasm/irony.
4. Select from these emotions: {[e.value for e in Emotion]}
5. Determine intensity ranging from [-1, 1]

## Output Format
```json
{{
  "emotion": "<emotion_name>",
  "intensity": <float>
}}
```

## Analysis Priority
1. **Text Analysis (Primary Focus):**
   - What emotion does the speaker actually express in this text?
   - Emotional vocabulary, tone, punctuation, emojis, and markers.
   - Intensity and clarity of emotional expression.

2. **Persona Consideration (Secondary, but Critical):**
   - How does the persona's character shape their emotional display?
   - Would this persona exaggerate, suppress, or distort certain emotions?
   - Adjust your judgment based on persona's typical emotional baseline and expression style.

3. **Context Reference (Tertiary):**
   - Use only to clarify ambiguous, sarcastic, or context-dependent emotions.
   - Reference recent conversation or situation if it changes the meaning of the text.

## Guidelines
1. **Emotion Selection:**
   - Choose from: {[e.value for e in Emotion]}
   - Focus on the dominant, most likely emotion the speaker is experiencing.
   - Only use "neutral" if the text is truly emotionless or ambiguous.

2. **Intensity Calculation:**
   - Consider word choice, expression strength, and emotional markers.
   - Adjust for persona's typical emotional range and expression habits.
   - Use context only if it clearly amplifies or dampens the emotion.

3. **Principles:**
   - Prioritize evidence from the text itself.
   - Let persona traits guide your interpretation of ambiguous or subtle emotions.
   - Use context to resolve uncertainty, not as the main basis.
   - Be objective and avoid over-interpretation.
"""
