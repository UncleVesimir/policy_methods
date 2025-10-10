from agents.REINFORCE_agent import REINFORCEAgent
from agents.actorCriticAgent import ActorCriticAgent

agents_dict = {
    "REINFORCE": REINFORCEAgent,
    "AC2": ActorCriticAgent
}

__all__ = ["REINFORCE", "AC2", "agents_dict"]