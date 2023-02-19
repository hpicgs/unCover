def normalize(doc: str) -> str:
    characters = [
        char.lower() for char in doc if char.isalnum() or char.isspace()
    ]
    return ''.join(characters)
