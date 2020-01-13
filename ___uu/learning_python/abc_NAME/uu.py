class ABCline:
    def joke(self):
        raise NotImplementedError()
    def punchline(self):
        raise NotImplementedError()

class BB(ABCline):
    def joke(self):
        print('A nice guy')


bb = BB()
bb.joke()