import json

def loadDict(file_path):
    """ Load Json dictionary data.
    Args:
        file_path: input Json file (.json).

    Returns:
        data: dictionary data.
    """
    with open(file_path, 'r') as f:
        json_data = f.read()
        data = json.loads(json_data)
        return data
    return None


def saveDict(file_path, data):
    """ Save the dictionary data to the given json file.

    Args:
        file_path: output json file path.
        data: data: dictionary data.
    """
    with open(file_path, 'w') as f:
        f.write(json.dumps(data))
