import os

from vision_models.core import PROJECT_ROOT


def resource_path(path: str, base_path = PROJECT_ROOT) -> str:
    return os.path.join(base_path, path)
