import random as random

import mesa
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector


class Obstacle(mesa.Agent):  # noqa
    def __init__(self, unique_id, model):
        """
        Customize the agent
        """
        self.unique_id = unique_id
        super().__init__(unique_id, model)

        self.role = "obstacle"
        self.intent = "obstacle"
        self.steal = "obstacle"
        self.object = "obstacle"
        self.drop = "obstacle"

    def step(self):
        #print("im a stone im blocking")
        pass


class AlleyAgent(mesa.Agent):  # noqa
    """
    An agent
    """
    def __init__(self, unique_id, model):
        """
        Customize the agent
        """
        self.unique_id = unique_id
        super().__init__(unique_id, model)

    def set_goal(self, goal):
        self.goal = goal

    def set_role(self, role):
        self.role = role

    def set_intent(self, intent):
        self.intent = intent

    def set_steal(self, steal):
        self.steal = steal

    def set_own_object(self, object):
        self.object = object

    def set_drop(self, dropped):
        self.drop = dropped

    def check_position(self, step):
        pos = step
        if self.model.obstacle_location!= False:
            if pos == self.model.obstacle_location:
                return False

        #if self.role == "thief":
        #    if pos == (0,1):
        #        return False

        # put obstacle here
        return True  # you can move anywhere else

    def move_step_to_goal(self):
        hx, hy = self.goal
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True, include_center=False)
        bestx, besty = 100, 100
        best = 100
        for step in possible_steps:
            stepx, stepy = step
            if ((hx - stepx) ** 2 + (hy - stepy) ** 2 < best) and self.check_position(step):
                best = (hx - stepx) ** 2 + (hy - stepy) ** 2
                bestx, besty = stepx, stepy

        if (bestx, besty) == (100, 100):
            possible_steps = self.model.grid.get_neighborhood(
                self.pos,
                moore=True,
                include_center=False)
            new_position = self.random.choice(possible_steps)
            self.model.grid.move_agent(self, new_position)
        else:
            self.model.grid.move_agent(self, (bestx, besty))

    def move(self):
        self.move_step_to_goal()



    def step(self):
        """
        Modify this method to change what an individual agent will do during each step.
        Can include logic based on neighbors states.

        """
        #print(self.unique_id, self.model.schedule.time, self.pos, self.goal)
        self.move()
        if self.role == "thief":
            self.goal = self.model.alleyagent_list[0].pos   # the thief wants to move to the victim
            for ob in self.model.grid.get_neighbors(self.pos, 1, True):
                if type(ob) == AlleyAgent:
                    if ob.role == "potentialvictim":
                        if ob.object == True and random.random() < self.model.thief_success_rate:
                            self.set_steal(True)
                            self.set_own_object(True)
                            ob.set_own_object(False)
                self.set_goal((0, self.model.grid.height - 1))


        if self.role == "potentialvictim":
            if random.random() < self.model.drop_rate and self.object == True:
                self.set_own_object(False)
                self.set_drop(True)
                #self.model.alleyagent_list[1].set_intent(False) # if dropped the thief does not have intent anymore





class AlleyModel(mesa.Model):
    """
    The model class holds the model-level attributes, manages the agents, and generally handles
    the global level of our model.

    There is only one model-level parameter: how many agents the model contains. When a new model
    is started, we want it to populate itself with the given number of agents.

    The scheduler is a special model component which controls the order in which agents are activated.
    """

    def __init__(self, num_agents, width, height, param_dict):
        super().__init__()
        self.num_agents = num_agents
        self.steal_threshold = param_dict["steal_threshold"] #0.7
        self.thief_success_rate = param_dict["thief_success_rate"] #0.9
        self.drop_rate = param_dict["drop_rate"] #0.3
        self.obstacle_rate = param_dict["obstacle_rate"] #0.6
        self.schedule = mesa.time.RandomActivation(self)
        self.grid = mesa.space.MultiGrid(width=width, height=height, torus=False)
        self.alleyagent_list = []

        for i in range(self.num_agents):

            agent = AlleyAgent(i, self)
            self.schedule.add(agent)
            if i == 0: # victim
                x = 1
                y = self.grid.height-1
                agent.set_goal((1, 0))
                agent.set_intent(False)
                agent.set_role("potentialvictim")
                agent.set_own_object(True)
                agent.set_steal(False)
                agent.set_drop(False)
            else: # thief
                x = 0
                y=0
                agent.set_intent(random.random())
                if agent.intent > self.steal_threshold:
                    agent.set_intent(True)
                    agent.set_role("thief")
                    agent.set_goal(self.alleyagent_list[0].pos)
                else:
                    agent.set_intent(False)
                    agent.set_role("innocent")
                    agent.set_goal((0,self.grid.height-1))
                agent.set_own_object(False)
                agent.set_steal(False)
                agent.set_drop(False)

            self.grid.place_agent(agent, (x, y))
            self.alleyagent_list.append(agent)

        if random.random()<self.obstacle_rate:
            obstacle = Obstacle(self.num_agents+1, self)
            self.schedule.add(obstacle)
            self.grid.place_agent(obstacle, (0, 1))
            self.obstacle_location = (0, 1)
        else:
            self.obstacle_location = False

        self.obstacle_evidence = self.get_obstacle_ev()

        # example data collector
        self.datacollector = mesa.datacollection.DataCollector(model_reporters={"steal_threshold":"steal_threshold",
                                                                                "thief_success_rate":"thief_success_rate",
                                                                                "drop_rate":"drop_rate",
                                                                                "Obstacle": lambda m:m.get_obstacle(),
                                                                                "Eobstacle":"obstacle_evidence",
                                                                                "SamePlace":lambda m:m.get_samePlace()},
                                                               agent_reporters={"step":lambda a:a.model.schedule.time,
                                                                                "id":"unique_id",
                                                                                "pos":"pos", "intent":"intent",
                                                                                "role":"role", "steal":"steal",
                                                                                "object":"object", "drop": "drop"})

        self.running = True
        self.datacollector.collect(self)
        self.all_agents_reached_goals = False

    def get_obstacle(self):
        if self.obstacle_location != False:
            return True
        else:
            return False

    def get_obstacle_ev(self):
        if self.obstacle_location != False: # the obstacle is present
            if random.random() < 0.8:
                return True # we see the obstacle
            else:
                return False # we do not see it even though it was there
        else:   # the obstacle is not there
            if random.random() < 0.1:
                return True # we see the obstacle, even though it is not there
            else:
                return False # we find no evidence for obstacle correctly, since it was not there

    def get_samePlace(self):
        if self.alleyagent_list[0].pos == self.alleyagent_list[1].pos:
            return str(True)
        else:
            return str(False)

    def step(self):
        """
        A model step. Used for collecting data and advancing the schedule
        """

        self.datacollector.collect(self)
        self.schedule.step()

        self.all_agents_reached_goals = True
        for agent in self.alleyagent_list:
            if agent.goal != agent.pos:
                self.all_agents_reached_goals = False
        if self.all_agents_reached_goals == True:
            self.datacollector.collect(self)
            self.running = False