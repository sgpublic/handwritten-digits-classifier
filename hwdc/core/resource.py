import os

from hwdc.core import PROJECT_ROOT


def hwdc_path(path: str) -> str:
    return os.path.join(PROJECT_ROOT, path)
