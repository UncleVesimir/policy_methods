from agents.REINFORCE_agent import REINFORCEAgent

agents_dict = {
    "REINFORCE": REINFORCEAgent,
}

__all__ = ["REINFORCE", "agents_dict"]