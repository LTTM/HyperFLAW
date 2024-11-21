from pathlib import Path


# Define the export
__all__ = ['load_paths_from_txt']


def load_paths_from_txt(txt_file_path):
    with open(txt_file_path, 'r') as f:
        paths = f.read().splitlines()
    return [Path(p) for p in paths]
