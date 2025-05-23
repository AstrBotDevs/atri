import json
import re


def parse_json(text: str) -> dict:
    """Parse a JSON object from the given text.

    Args:
        text (str): The text to parse.

    Returns:
        dict: The parsed JSON object.
    """
    # Remove the code block markers
    if """```""" not in text:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}
    pattern = re.compile(r"(?i)```json[\s\r\n]*(.*?)```", re.DOTALL)
    matches = pattern.findall(text)
    return json.loads(matches[0]) if matches else {}


# test
if __name__ == "__main__":
    text = """
    123
    ```json
    {
        "key": "value"
    }
    ```
    aocenaov
    """
    import asyncio
    result = asyncio.run(parse_json(text))
    print(result)  # {'key': 'value'}
