import os
import json
import ast
import logging
import types
import contextlib
from contextlib import contextmanager
from balrog.agents.base import BaseAgent
from balrog.client import LLMResponse
from collections import namedtuple
from balrog.client import LLMClientWrapper
from functools import wraps
import sys

from . import adas_prompts
import traceback

from ..prompt_builder.history import Message 

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class AdasAgent(BaseAgent):

    def __init__(self, client_factory, prompt_builder, env_name, lock, output_dir, eval_mode, config):
        super().__init__(client_factory, prompt_builder)

        self.config = config
        self.client_factory = client_factory
        # Create client with json structure output turned on
        self.meta_client = type(self.client)(self.config.client, json_structure=True) 
        # self.meta_client = self.client
        self.prompt_builder = prompt_builder
        self.meta_prompt_builder = [Message(role="system", 
                                            content="You are a helpful assistant. Make sure to return in a WELL-FORMED JSON object.")] 
        self.env_name = env_name
        self.lock = lock if lock else contextlib.nullcontext()
        self.output_dir = output_dir
        self.eval_mode = eval_mode
        self.error_budget = config.adas.error_budget
        self.archive_path = os.path.join(self.output_dir, 
                                         config.adas.agents_archive, 
                                         self.env_name)
        self.eval_agent_path = self.config.adas.agents_from
        self.archive = []
        self.base_agents = _extract_agents(self.config.adas.base_agents_path)

        self.agent = None
        self.agent_code = None

    def catch_llm_error(func):
        @wraps(func)
        def subject_function(self, candidate_agent):
            while self.error_budget > 0:
                try:
                    logger.info("Inside of catch_llm_error, budget left: " + str(self.error_budget))
                    return func(self, candidate_agent), candidate_agent
                except Exception as e:
                    error_traceback = traceback.format_exc()
                    current_directory = os.getcwd()
                    anonymized_traceback = error_traceback.replace(current_directory + os.path.sep, './')
                    anonymized_traceback = anonymized_traceback.replace(current_directory, '.')
                    print("--------------------------------")
                    print("with LLM output: \n", candidate_agent, "\n\n Ran into error: \n", anonymized_traceback)
                    print("--------------------------------")
                    with self.lock:
                        self.error_budget -= 1
                        candidate_agent, self.meta_prompt_builder = _error_reflect(self.meta_client, self.meta_prompt_builder, candidate_agent, anonymized_traceback)
            raise RuntimeError("Ran out of error budget")
        return subject_function

    def init_with_obs(self, init_obs):
        # check if agents have been given to evaluate
        if self.eval_mode and self.config.eval.agents_from == null:
            raise NameError("Unable to evaluate agents without path given, please update config.")

        with self.lock:
            if self.eval_mode:
                if not os.path.exists(self.eval_agent_path): raise NameError("Could not find given agent_from dir from config.")
                if len(os.listdir(self.eval_agent_path)) == 0: raise NameError("Given agent_from dir is empty.")
                # set agent to best performing agent from archive
                self.agent_code = max(_extract_agents(self.eval_agent_path), key=lambda d: d['score'])
                self.agent, self.agent_code = exec_agent(self.agent_code)

            elif os.path.exists(os.path.join(self.archive_path, "current_agent.json")):
                self.agent = self.load_agent_from_file()
            else:
                # not in eval_mode mode, load all .jsons, generate new one, refine it twice
                self.archive = _extract_agents(self.archive_path, self.lock)        

                # If the archive_path does not exist (not agents generated yet) generate agent
                candidate_agent, self.meta_prompt_builder = _generate_agent(self.meta_client,
                                                                            self.meta_prompt_builder,
                                                                            self.prompt_builder.system_prompt, 
                                                                            init_obs,
                                                                            self.archive + self.base_agents)
                candidate_agent, self.meta_prompt_builder = _self_reflection_one(self.meta_client, self.meta_prompt_builder, candidate_agent)
                candidate_agent, self.meta_prompt_builder = _self_reflection_two(self.meta_client, self.meta_prompt_builder, candidate_agent)

                self.agent_code, _ = self.load_json(candidate_agent)
                self.agent, self.agent_code = self.exec_agent(self.agent_code)
                self.save_agent_to_file(self.agent_code)
    
    @catch_llm_error
    def exec_agent(self, agent_code):
        logger.info("Inside of exec_agent")
        staging_namespace = {}
        agent_code, _ = self.load_json(agent_code) if isinstance(agent_code, str) else (agent_code, 0) 
        exec(agent_code['code'], globals(), staging_namespace)
        new_class_name = agent_code['name']
        globals().update(staging_namespace) # merge namespaces to allow for imports
        return staging_namespace[new_class_name](self.client_factory, self.prompt_builder)

    @catch_llm_error
    def load_json(self, json_string):
        logger.info("Inside of load_json...")
        # json_string = json_string[json_string.index('{'):json_string.rindex('}')+1]
        return ast.literal_eval(json_string)

    def load_agent_from_file(self):
        with self.lock, open(os.path.join(self.archive_path, "current_agent.json"), "r") as f:
           return json.load(f)

    def save_agent_to_file(self, agent_code):
        logger.info("inside of save_agent_to_file...")
        with self.lock:
            os.makedirs(self.archive_path, exist_ok=True)
            with open(os.path.join(self.archive_path, "current_agent.json"), "w+") as f:
                json.dump(agent_code, f) 
        logger.info("finished saving agent to file")

    def act(self, obs, prev_action=None):
        # initialize agent if agent doesn't exist yet
        if self.agent is None:
            self.init_with_obs(obs)

        if self.eval_mode:
            # acting in eval_mode, don't fix if there's errors
            try:
                return self.agent.act(obs, prev_action)
            except Exception:
                return ""
        else:
            # first, reflect on if past agent had errors
            if "Your previous output did not contain a valid action." in obs['text']:
                print("Failed to get valid action")
                print("Obs: \n", obs)
                source_code, self.meta_prompt_builder = _error_reflect(self.meta_client, self.meta_prompt_builder, 
                                                                     self.agent_code, "Your previous output did not contain a valid action.") 
                self.agent_code, _ = self.load_json(source_code)
            else:
                print("didn't find error, here is the obs: ", obs)
                self.agent_code = self.load_agent_from_file()
            self.agent, self.agent_code = self.exec_agent(self.agent_code)
            self.save_agent_to_file(self.agent_code)
            while self.error_budget > 0:
                try:
                    return self.agent.act(obs, prev_action)
                except Exception as e: 
                    self.error_budget -= 1
                    error_traceback = traceback.format_exc()
                    current_directory = os.getcwd()
                    anonymized_traceback = error_traceback.replace(current_directory + os.path.sep, './')
                    anonymized_traceback = anonymized_traceback.replace(current_directory, '.')
                    source_code, self.meta_prompt_builder = _error_reflect(self.meta_client, self.meta_prompt_builder,
                                                                           self.agent_code, anonymized_traceback)
                    self.agent_code, _ = self.load_json(source_code)
                    self.agent, self.agent_code = self.exec_agent(self.agent_code)
                    self.save_agent_to_file(self.agent_code)
            raise RuntimeError("Ran out of error budget in act")


def _extract_agents(agent_dir, lock=contextlib.nullcontext()):
    if agent_dir is None or not os.path.exists(agent_dir):         
        print(f"Could not find path {agent_dir} empty array returned.")
        return [] 
    with lock:
        return [json.load(open(os.path.join(agent_dir, agent_path), "r")) 
                for agent_path in os.listdir(agent_dir) if agent_path[-5:] == ".json"]

def _generate_agent(client, meta_prompt_builder, system_prompt, init_obs, archive):
    logger.info("Inside of _generate_agent...")
    full_prompt = Message(role="user", content=
                                        adas_prompts.prompt_instruction 
                                + "\n" + system_prompt 
                                + "\nn" +  "An example observation is given below:"
                                + "\n\n" + str(init_obs)
                                + "\n\n" + adas_prompts.framework_instructions
                                + "\n\n" + str(archive)
                                + "\n\n" + adas_prompts.task_instruction
                                + "\n\n" + adas_prompts.output_instruction
                                + "\n\n" + adas_prompts.wrong_impl)
    meta_prompt_builder += [full_prompt]
    new_agent = client.generate(meta_prompt_builder)
    logger.info("Generated:")
    logger.info(new_agent.completion)
    logger.info("-----------------------------")
    return new_agent.completion, meta_prompt_builder

def _self_reflection_one(client, meta_prompt_builder, agent):
    logger.info("Inside of _self_reflection_one...")
    reflect_prompt = adas_prompts.reflection_one
    reflect_message = Message(role="user", content=str(agent) + "\n" + reflect_prompt)
    meta_prompt_builder += [reflect_message]
    new_agent = client.generate(meta_prompt_builder)
    logger.info("Generated:")
    logger.info(new_agent.completion)
    logger.info("----------------------------")
    return new_agent.completion, meta_prompt_builder

def _self_reflection_two(client, meta_prompt_builder, agent):
    logger.info("Inside of _self_reflection_two...")
    reflect_prompt = adas_prompts.reflection_two
    reflect_message = Message(role="user", content=str(agent) + "\n" + reflect_prompt)
    meta_prompt_builder += [reflect_message]
    new_agent = client.generate(meta_prompt_builder)
    logger.info("Generated:")
    logger.info(new_agent.completion)
    logger.info("---------------------------")
    return new_agent.completion, meta_prompt_builder

def _error_reflect(client, meta_prompt_builder, agent, error):
    logger.info("Inside of _error_reflect...")
    error_prompt = adas_prompts.error_prompt(error)
    error_message = Message(role="user", content=str(agent) + "\n" + error_prompt)
    meta_prompt_builder += [error_message]
    new_agent = client.generate(meta_prompt_builder)
    logger.info("Generated:")
    logger.info(new_agent.completion)
    logger.info("--------------------------")
    return new_agent.completion, meta_prompt_builder


"""
Need to:
    - [?] make sure eval.py works with current format
    - [X] modify this file to do everything but with locks (if a lock is given meaning multithreading)
    - [X] wrong impl list
    - [X] modify the __init__.py to pass locks and env_name
    - [X] modify the evaluator file to pass locks and env_name
    - [X] modify the config to have all the information about where to put agents, what to run when in 'training' mode (determined by generating_adas.py not config)
    - [X] create util function that modifies the config per adas episode
    - [ ] create util function that saves the agents
    - [ ] write generate_adas.py file to create the agents
    - [X] implement base agents function
"""

