"""
Configure visualization elements and instantiate a server
"""

import mesa
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.modules import TextElement
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.ModularVisualization import VisualizationElement
from alley.alley.model import AlleyModel, AlleyAgent, Obstacle  # noqa
import random as random

def circle_portrayal_example(agent):

    print(agent)
    if agent is None:
        return

    if type(agent) == Obstacle:

        portrayal = {
            "Shape": "circle",
            "Filled": "true",
            "Layer": 0,
            "r": 0.5,
            "Color": random.choice(["Gainsboro"])
        }

    else:
        portrayal = {
            "Shape": "circle",
            "Filled": "true",
            "Layer": 0,
            "r": 0.5,
        }
        if agent.role == "thief":
            portrayal["Color"]= random.choice(["Purple"])
        elif agent.role == "innocent":
            portrayal["Color"]= random.choice(["Goldenrod"])
        else:
            portrayal["Color"]="Orange"

    return portrayal


grid = CanvasGrid(circle_portrayal_example, 2, 3, 500, 500)

model_kwargs = {"num_agents": 2, "width": 2, "height": 3}

server = ModularServer(AlleyModel,
[grid],
"My Model", {"num_agents": 2, "width": 2, "height": 3,
             "param_dict":{"steal_threshold": 0.7, "thief_success_rate": 0.9,
             "drop_rate":0.3, "obstacle_rate":0.6}})

