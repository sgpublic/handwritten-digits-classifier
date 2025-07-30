import os

from hwdc.core import PROJECT_ROOT


def hwdc_path(path: str, base_path = PROJECT_ROOT) -> str:
    return os.path.join(base_path, path)
