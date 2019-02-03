from abc import ABCMeta
from abc import ABC
from abc import abstractmethod
# class Comedian(metaclass=ABCMeta):
class Comedian_Degree(ABC):
    @abstractmethod
    def joke(self):
        print('Let me tell you a joke:')
        # raise NotImplementedError()
    @abstractmethod  # with or without
    def punchline(self):
        pass

class Student(Comedian_Degree):
    def joke(self):
        super().joke()
        print("You don't need a parachute to go skydiving.")
    def punchline(self):
        print("You need a parachute to go skydiving twice.")

comedian = Student()
comedian.joke()