from enum import Enum


class ModelSaveType(Enum):
    ORIGIN = "pth"
    ONNX = "onnx"

    def with_file_name(self, file_name: str) -> str:
        return f"{file_name}.{self.value}"
