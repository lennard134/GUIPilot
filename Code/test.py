import re
def map_to_snli_label(response):
    """
    Extract the SNLI label ('entailed' or 'not-entailed') from a model's response.
    """
    # Normalize response
    response = response.lower().strip()

    # Prioritize explicit conclusions, capturing common conclusion phrases
    conclusion_patterns = [
        r"Conclusion: (entailed|not-entailed)",
        r"the relationship between the premise and hypothesis is ['\"]?(entailed|not-entailed)['\"]?",
        r"reasoning-step conclusion:\s*(entailed|not-entailed)", 
        r"therefore, the hypothesis is ['\"]?(entailed|not-entailed)['\"]?", 
        r"the hypothesis (must be|is)\s+(entailed|not-entailed)",
    ]

    for pattern in conclusion_patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()

    # Fallback check for the explicit labels appearing in isolation
    if "not-entailed" in response:
        return "not-entailed"
    elif "entailed" in response:
        return "entailed"

    # If no label is found, return None to indicate no conclusion detected
    return None

if __name__ == "__main__":
    test_query = 'the hypothesis is not entailed from the premis'
    print(map_to_snli_label(test_query))