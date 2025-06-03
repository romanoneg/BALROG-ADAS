import os
import importlib

from balrog.agents.base import BaseAgent
from balrog.client import LLMClientWrapper

class AdasAgent():

    def __init__(self, client_factory: LLMClientWrapper, prompt_builder, 
                 original_cwd="", output_dir=".", env_name: str, episode_idx: int, config):
        super().__init__(client_factory, prompt_builder)
        
        base_path = os.join(self.output_dir,
                                  config.eval.agents_dir,
                                  env_name)

        prev_agent_filename = os.join(base_path, f"{env_name}_run_{max(episode_idx - 1, 0):02d}.py")
        agent_filename = os.join(base_path, f"{env_name}_run_{episode_idx:02d}.py")

        if os.path.exists(prev_agent_filename):
            # TODO: add logging of errors
            with open(prev_agent_filename, "r") as f:
                source_code = f.read()

            # stagging_namespace = {}
            # exec(source_code, globals(), stagging_namespace)
            #
            # if "act" in stagging_namespace:
            #     self.act = stagging_namespace['act']
            # else:
            #     raise NameError(f"Couldn't find the 'act' function in {prev_agent_filename}")

        return

