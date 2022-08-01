from abc import ABC, abstractmethod
import json
import sys, os
sys.path.append(os.path.join(sys.path[0], '../'))

class PathProvider(ABC):

    @staticmethod
    @abstractmethod
    def board() -> str: pass

    @staticmethod
    @abstractmethod
    def models() -> str: pass

    @staticmethod
    @abstractmethod
    def data() -> str: pass


class DGXPath(PathProvider):
    pathes = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "./pathes.json")))

    @staticmethod
    def board() -> str:
        return DGXPath.pathes['tensorboard']

    @staticmethod
    def models() -> str:
        return DGXPath.pathes['models']

    @staticmethod
    def data() -> str:
        return DGXPath.pathes['data']


class Paths:

    default: PathProvider = DGXPath







