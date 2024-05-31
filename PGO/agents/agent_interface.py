from abc import ABC, abstractmethod

class Agent(ABC):
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def plot(self):
        pass

    @abstractmethod
    def test(self):
        pass