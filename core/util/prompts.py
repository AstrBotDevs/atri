SUMMARIZE_PROMPT = """
You are a expert of summarizing chats.
You will be given a chats between a user and an assistant.

## Requirement
- Your task is to summarize the chats into 1~2 sentences.
- Only extract the most critical events that occurred.
- Time and Schedule should be included in the summary if possible.
- Only return the plain summary and no explanation.

The summary should use the same language as the chats.
"""

EXTRACT_ENTITES_PROMPT = """
You are a expert of extracting entities from text.
You will be given a text which is a summary of a chat between a user and an assistant, You need to extract the entities from the summarization.

## Examples:
- User's Input: "Json loves to play basketball with his friends on weekends."
  Your Output:
```json
{
    "entities": [
        {
            "type": "person",
            "name": "John"
        },
        {
            "type": "activity",
            "name": "basketball"
        },
        {
            "type": "time",
            "name": "weekends"
        }
    ]
}

## Requirements
- You must use the same language as user's input.
- You should use `USER_ID` as the entity name for any self-references (e.g., "I", "me", "my" etc.) in user messages.
```
"""

BUILD_RELATIONS_PROMPT = """
You are a expert of building relations between entities.
You will be given a list of entities extracted from the text and the original text, You need to build relations between them.

## Examples:
- User's Input: "Json loves to play basketball with his friends on weekends."
  Your Output:
```json
{
    "relations": [
        {
            "source": "Json",
            "target": "basketball",
            "relation_type": "likes"
        },
        {
            "source": "Json",
            "target": "friends",
            "relation_type": "plays_with"
        },
        {
            "source": "Json",
            "target": "weekends",
            "relation_type": "plays_on"
        }
    ]
}

## Requirements
- You must use the same language as user's input.
- You should use `USER_ID` as the source or target content for any self-references (e.g., "I", "me", "my" etc.) in user messages.

```
"""
