class Command:
    def do(self): ...
    def undo(self): ...

class UndoStack:
    def __init__(self, limit: int = 500):
        self._stack = []
        self._redo = []
        self._limit = limit

    def push(self, cmd: Command):
        cmd.do()
        self._stack.append(cmd)
        self._redo.clear()
        if len(self._stack) > self._limit:
            self._stack.pop(0)

    def undo(self):
        if not self._stack:
            return
        cmd = self._stack.pop()
        cmd.undo()
        self._redo.append(cmd)

    def redo(self):
        if not self._redo:
            return
        cmd = self._redo.pop()
        cmd.do()
        self._stack.append(cmd)
