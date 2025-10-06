from alley.alley.server import server  # noqa
from alley.alley.model import AlleyModel


def run_model(param_dict):
    model = AlleyModel(2, 2, 3, param_dict=param_dict)
    for i in range(0, 3):
        model.step()
    return model

def run_visual():
    server.launch()

#run_visual()