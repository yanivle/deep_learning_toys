
class Model:
    def __init__(self) -> None:
        self.params = []
        self.params_names = []
    
    def addParam(self, name, param):
        self.params.append(param)
        self.params_names.append(name)
        setattr(self, name, param)

    def __repr__(self) -> str:
        params_str = '\n  '.join([f'{name}({param.size} parameters)' for param, name in zip(self.params, self.params_names)])
        return f'{type(self).__name__}({self.num_params} parameters):\n  {params_str}'

    @property
    def num_params(self):
        return sum(param.size for param in self.params)
