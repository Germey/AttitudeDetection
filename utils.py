import json


def load_tensors(path):
    """
    load tensors from json file
    :param path: tensor file path
    :return: tensors
    """
    with open(path, encoding='utf-8') as f:
        return json.loads(f.read())
