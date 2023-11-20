from iml import IML as iiIML

class IML:
    def __init__(self, context, **kwargs) -> None:
        self.ctx = context
        self.kwargs = kwargs
        self.iml = iiIML(**kwargs)
