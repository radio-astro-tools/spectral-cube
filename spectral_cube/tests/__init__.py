import os


def path(filename):
    return os.path.join(os.path.dirname(__file__), 'data', filename)
