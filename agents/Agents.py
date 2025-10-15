from agents.REINFORCE_agent import REINFORCEAgent
from agents.actorCriticAgent import ActorCriticAgent

agents_dict = {
    "REINFORCE": REINFORCEAgent,
    "A2C": ActorCriticAgent
}

__all__ = ["REINFORCE", "A2C", "agents_dict"]