class GeneralException(RuntimeError):
    def __init__(self, arg):
        self.args = [arg]


class ToolsException(RuntimeError):

    def __init__(self, arg):
        self.args = [arg]


class DiarizationException(RuntimeError):

    def __init__(self, arg):
        self.args = [arg]


class CalibrationIvecException(RuntimeError):

    def __init__(self, arg):
        self.args = [arg]


class CalibrationIvecSetException(RuntimeError):

    def __init__(self, arg):
        self.args = [arg]


class EvaluationIvecException(RuntimeError):

    def __init__(self, arg):
        self.args = [arg]


class EvaluationIvecSetException(RuntimeError):

    def __init__(self, arg):
        self.args = [arg]


class IvecException(RuntimeError):

    def __init__(self, arg):
        self.args = [arg]


class ClassifierException(RuntimeError):

    def __init__(self, arg):
        self.args = [arg]


class NormalizationException(RuntimeError):

    def __init__(self, arg):
        self.args = [arg]


