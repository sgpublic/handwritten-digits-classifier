from abc import abstractmethod, ABC
from logging import Logger

from vision_models.core.utils.logger import create_logger


class Log(ABC):
    def __init__(self):
        self._logger = create_logger(self.__logger_name__)

    @property
    def logger(self) -> Logger:
        return self._logger

    @property
    @abstractmethod
    def __logger_name__(self) -> str:
        pass
