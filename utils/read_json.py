import json


def read_left_eye_opened(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data['ObjectInfo']['BoundingBox']['Leye']['Opened']


def read_right_eye_opened(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data['ObjectInfo']['BoundingBox']['Reye']['Opened']


def read_file_name(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data['FileInfo']['FileName']


def read_key_points(path):
    with open(path, 'r') as f:
        data = json.load(f)

    if data['ObjectInfo']['KeyPoints']['Count'] > 0:
        return data['ObjectInfo']['KeyPoints']['Points']
