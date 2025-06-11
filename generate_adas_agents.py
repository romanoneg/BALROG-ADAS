import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from balrog.agents import AgentFactory
from balrog.evaluator import EvaluatorManager
from balrog.utils import collect_and_summarize_results, print_summary_table, setup_environment


@contextmanager
def redirect_to_file(filepath):
    original = sys.stdout
    with open(filepath, "w") as file:
        sys.stdout = file
        try:
            yield
        finally:
            sys.stdout = original

def save_agent(agent_factory, results):

    for env_name in results.keys():

        base_path = os.path.join(agent_factory.output_dir, env_name,)
        cur_agent_path = os.path.join(base_path, "current_agent.json")

        # read full json file
        # add the score to that json
        # write to the current_agent file

        os.rename(cur_agent_path, os.path.join(base_path, name))

# NOTE: Changed config name
@hydra.main(config_path="balrog/config", config_name="config", version_base="1.1")
def main(config: DictConfig):
    original_cwd = get_original_cwd()
    setup_environment(original_cwd=original_cwd)

    # Determine output directory
    if config.eval.resume_from is not None:
        output_dir = config.eval.resume_from
    else:
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{timestamp}_{config.agent.type}_{config.client.model_id.replace('/', '_')}"
        output_dir = os.path.join(config.eval.output_dir, run_name)

        # Create the directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Setup logger
    log_filename = os.path.join(output_dir, "eval.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filename)],
        force=True,
    )

    # Create AgentFactory/LLM so we can start querying it
    # Create EvaluatorManager, create agents, run evals on agents
    adas_config = config.copy()
    adas_config.envs = adas_config.adas.envs
    adas_config.eval.num_episodes = adas_config.adas.num_env_episodes_per_training_episodes
    evaluator_manager = EvaluatorManager(adas_config, original_cwd=original_cwd, output_dir=output_dir)
    agent_factory = AgentFactory(config, output_dir=output_dir, eval_mode=False) # NOTE: eval mode is false to signify make agents

    for episode in range(config.adas.num_training_episodes):
        results = evaluator_manager.run(agent_factory)
        print("FINAL RESULTS: ", results)
        # NOTE: this is where the agent is actually saved
        # save_agent(agent_factory, results)

    # Create Full EvaluatorManager, run evals


    # -------------- Old Code ----------------
    # # Create an EvaluatorManager and run evaluation
    # evaluator_manager = EvaluatorManager(config, original_cwd=original_cwd, output_dir=output_dir)
    # agent_factory = AgentFactory(config)
    # with redirect_to_file(log_filename):
    #     evaluator_manager.run(agent_factory)
    #
    # # Collect and summarize results
    # summary = collect_and_summarize_results(output_dir)
    # print_summary_table(summary)
    #

if __name__ == "__main__":
    main()
