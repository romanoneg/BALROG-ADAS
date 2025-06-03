import copy
import csv
import json
import logging
import multiprocessing
import os
import random
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from balrog.utils import get_unique_seed
from balrog.evaluator import Evaluator 
from balrog.prompt_builder.history import Message

logger = logging.getLogger(__name__)


class AdasManager:
    """Takes care of generating the ADAS agents based on the config file.
    
    Similar to the EvaluatorManager initializes evaluators for each specified environment and handles the execution
    of evaluation tasks either sequentially or in parallel using multiple workers while iterating through agent
    generation prompts. Saves task specific agents in files from config.
    """

    def __init__(self, config, original_cwd="", output_dir="."):
        """Initialize the AdasManager.

        Args:
            config (omegaconf.DictConfig): Configuration object containing evaluation settings.
            original_cwd (str, optional): Original current working directory. Defaults to "".
            output_dir (str, optional): Directory to save evaluation outputs. Defaults to ".".
        """
        self.config = config
        self.original_cwd = original_cwd
        self.output_dir = output_dir

        self.env_names = config.envs.names.split("-")
        self.env_evaluators = {}
        # NOTE: change the tasks to be batched by episode
        self.episodic_tasks = []
        for env_name in self.env_names:
            evaluator = Evaluator(env_name, config, original_cwd=original_cwd, output_dir=self.output_dir)
            self.env_evaluators[env_name] = evaluator
            for task in evaluator.tasks:
                tasks = []
                for episode_idx in range(evaluator.num_episodes):
                    # Check if task has been completed
                    json_filename = os.path.join(
                        self.output_dir,
                        env_name,
                        task,
                        f"{task}_run_{episode_idx:02d}.json",
                    )
                    if os.path.exists(json_filename):
                        logging.info(f"Skipping completed task: {env_name}, {task}, episode {episode_idx}")
                    else:
                        tasks.append((env_name, task, episode_idx))
                self.episodic_tasks.append(tasks)
        self.num_workers = config.eval.num_workers

    def run(self, agent_factory: AdasAgentFactory):
        """Run the evaluation using the specified agent factory.

        Args:
            agent_factory (AgentFactory): Factory object to create agents for evaluation.

        Returns:
            dict: Results of the evaluation aggregated by environment name.
        """
        if self.num_workers > 1:
            results = self._run_parallel(agent_factory)
        else:
            results = self._run_sequential(agent_factory)
        return results

    def _run_sequential(self, agent_factory: AdasAgentFactory):
        """Run the evaluation sequentially.

        Args:
            agent_factory (AgentFactory): Factory object to create agents for evaluation.

        Returns:
            dict: Results of the evaluation aggregated by environment name.
        """
        # NOTE: results batched by episode now 
        total_results = [] 
        total_task_num = sum(len(episode_tasks) for episode_task in self.episodic_tasks) 
        with tqdm(total=total_task_num, desc="Evaluating Episodes", position=0) as pbar:
            for episode_tasks in self.episodic_tasks:
                results = defaultdict(list)
                for env_name, task, episode_idx in episode_tasks: 
                    evaluator = self.env_evaluators[env_name]
                    agent = agent_factory.create_agent(env_name=env_name, episode_idx=episode_idx)
                    episode_log = evaluator.run_episode(task, agent, position=1, episode_idx=episode_idx)
                    results[env_name].append(episode_log)
                    pbar.update(1)
                # TODO Run the ADAS STEP HERE:
                print("Running ADAS step for episode rewards...")
                agent_factory.adas_step(results, episode_idx)
                total_results.append(results)
        return total_results

    def _run_parallel(self, agent_factory: AdasAgentFactory):
        """Run the evaluation in parallel using multiple workers.

        Args:
            agent_factory (AgentFactory): Factory object to create agents for evaluation.

        Returns:
            dict: Results of the evaluation aggregated by environment name.
        """
        task_queue = multiprocessing.Queue()
        results_queue = multiprocessing.Queue()

        ctx = multiprocessing.get_context("fork")

        # Initially fill the task queue with tasks up to the number of workers
        for item in self.tasks[: self.num_workers]:
            task_queue.put(item)

        # Create a master progress bar
        pbar = tqdm(total=len(self.tasks), position=0, leave=True)

        # Assign unique positions for progress bars
        positions = list(range(self.num_workers))

        processes = []
        for idx in range(self.num_workers):
            position = positions[idx]
            p = ctx.Process(
                target=self._worker,
                args=(task_queue, results_queue, agent_factory, position),
            )
            processes.append(p)
            p.start()

        results = defaultdict(list)
        tasks_completed = 0
        tasks_queued = self.num_workers

        total_tasks = len(self.tasks)

        while tasks_completed < total_tasks:
            result = results_queue.get()
            if "error" in result:
                logging.error(f"Error in task {result['task']} processed by {result['process_num']}: {result['error']}")
                logging.error(f"Traceback:\n{result['traceback']}")
            else:
                results[result["env_name"]].append(result)
            tasks_completed += 1

            # Update progress bar
            pbar.update(1)
            pbar.set_description(f"Last task: {result['task']}, Process: {result.get('process_num', 'N/A')}")

            # Queue another task if there are any left
            if tasks_queued < total_tasks:
                task_queue.put(self.tasks[tasks_queued])
                tasks_queued += 1

        # Signal workers to stop
        for _ in range(self.num_workers):
            task_queue.put(None)

        # Wait for all processes to finish
        for p in processes:
            p.join()

        # Close the master bar when done
        pbar.close()

        return results

    def _worker(self, task_queue, results_queue, agent_factory, position):
        """Worker process for parallel evaluation.

        Args:
            task_queue (multiprocessing.Queue): Queue containing tasks to process.
            results_queue (multiprocessing.Queue): Queue to put the results.
            agent_factory (AgentFactory): Factory object to create agents.
            position (int): Position index for the progress bar.
        """
        seed = get_unique_seed(process_num=position)
        random.seed(seed)
        np.random.seed(seed)

        process_num = multiprocessing.current_process().name
        while True:
            item = task_queue.get()
            if item is None:
                break
            try:
                env_name, task, episode_idx = item
                evaluator = self.env_evaluators[env_name]
                agent = agent_factory.create_agent(env_name=env_name, episode_idx=episode_idx) # used in ADAS agents
                result = evaluator.run_episode(
                    task,
                    agent,
                    process_num=process_num,
                    position=position + 1,
                    episode_idx=episode_idx,
                )
                result["process_num"] = process_num  # Include process number in result
                result["env_name"] = env_name
                results_queue.put(result)
            except Exception as e:
                tb = traceback.format_exc()
                logging.error(f"Error in worker processing task {task}: {e}\n{tb}")
                results_queue.put(
                    {
                        "env_name": env_name,
                        "task": task,
                        "error": str(e),
                        "traceback": tb,
                        "process_num": process_num,
                    }
                )


class AdasAgentFactory(AgentFactory):
    """
    """

    def __init__(self, original_cwd="", output_dir=".", config):
        """
        """
        super().__init__(config)
        self.original_cwd = original_cwd
        self.output_dir = output_dir
        self.config = config

    def create_agent(self, env_name=None, episode_idx=None):
        """

        """
        client_factory = create_llm_client(self.config.client)
        prompt_builder = create_prompt_builder(self.config.agent)

        assert self.config.agent.type == "adas"
        return AdasAgent(client_factory, prompt_builder, env_name=env_name, episode_idx=episode_idx 
                             original_cwd=original_cwd, output_dir=output_dir, config=self.config)

    def run_adas_step(self, results, episode_idx):

        client = create_llm_client(self.config.client)
        
        for env_name in results.keys():

            agentdir_path = os.path.join(self.output_dir,
                                      self.config.eval.agents_dir, 
                                      env_name, 
                                      f"{env_name}_run_{episode_idx:02d}")

            # Read all agents from agent dir
            agent_code = []
            for filename in os.listdir(agentdir_path):
                agent_path = os.join(agentdir_path, filename)
                with open(agent_path, "r") as f:
                    agent_code.append(f.read())






