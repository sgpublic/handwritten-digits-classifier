from hwdc.core.resource import hwdc_path


class HwdcModel:
    def __init__(self,
                 save_path: str = hwdc_path("./model")):
        self._save_path = save_path

    def load(self):
        pass

    def save(self):
        pass
