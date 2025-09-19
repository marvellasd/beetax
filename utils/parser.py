import re
import json

def extract_json_dict(text):
    """
    Extracts and parses the JSON dictionary from the input text.

    Args:
        text (str): The input text containing a JSON object inside {}.

    Returns:
        dict: The parsed JSON dictionary.

    Raises:
        ValueError: If no valid JSON object is found or if parsing fails.
    """
    # Use regex to find the JSON object enclosed in {}
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        print("No match found in the text.")
        return None

    # Extract the JSON string
    json_str = match.group(0)

    json_str = re.sub(r"\\'", "'", json_str)

    try:
        # Parse the JSON string into a Python dictionary
        json_dict = json.loads(json_str)
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)
        return None

    return json_dict