import signal
from typing import Any


class CatchSIGTERM(Exception):
    @staticmethod
    def __str__() -> str:
        return "Exception caused by catching SIGTERM"

    @staticmethod
    def __repr__() -> str:
        return "CatchSIGTERM()"

    @staticmethod
    def handler(*_: Any) -> None:
        raise CatchSIGTERM

    @classmethod
    def start_signal_handling(cls) -> None:
        signal.signal(signal.SIGTERM, cls.handler)
