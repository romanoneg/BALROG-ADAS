from balrog.client import create_llm_client

from ..prompt_builder import create_prompt_builder
from .chain_of_thought import ChainOfThoughtAgent
from .custom import CustomAgent
from .dummy import DummyAgent
from .few_shot import FewShotAgent
from .naive import NaiveAgent
from .robust_naive import RobustNaiveAgent
from .robust_cot import RobustCoTAgent
from .adas import AdasAgent

class AgentFactory:
    """Factory class for creating agents based on configuration.

    The `AgentFactory` class is responsible for initializing the appropriate agent type
    based on the provided configuration, which includes setting up the LLM client and
    prompt builder.
    """

    def __init__(self, config, output_dir="."):
        """Initialize the AgentFactory with configuration settings.

        Args:
            config (omegaconf.DictConfig): Configuration object containing settings for the agent and client.
        """
        self.config = config
        self.output_dir = output_dir

    def create_agent(self, env_name=None):
        """Create an agent instance based on the agent type specified in the configuration.

        The function uses the `config.agent.type` attribute to determine which agent to create.
        It supports several agent types, including Naive, Chain-of-Thought, Self-Refine, Dummy,
        and Custom agents.

        Args:
            env_name: Used to select corresponding agent for enviornment in ADAS mode. Does not affect
                      other agent types.

        Returns:
            Agent: An instance of the selected agent type, configured with the client and prompt builder.

        Raises:
            ValueError: If an unknown agent type is specified in the configuration.
        """
        client_factory = create_llm_client(self.config.client)
        prompt_builder = create_prompt_builder(self.config.agent)

        if self.config.agent.type == "naive":
            return NaiveAgent(client_factory, prompt_builder)
        elif self.config.agent.type == "cot":
            return ChainOfThoughtAgent(client_factory, prompt_builder, config=self.config)
        elif self.config.agent.type == "dummy":
            return DummyAgent(client_factory, prompt_builder)
        elif self.config.agent.type == "custom":
            return CustomAgent(client_factory, prompt_builder)
        elif self.config.agent.type == "few_shot":
            return FewShotAgent(client_factory, prompt_builder, self.config.agent.max_icl_history)
        elif self.config.agent.type == "robust_naive":
            return RobustNaiveAgent(client_factory, prompt_builder)
        elif self.config.agent.type == "robust_cot":
            return RobustCoTAgent(client_factory, prompt_builder, config=self.config)
        elif self.config.agent.type == "adas":
            return AdasAgent(client_factory, prompt_builder, env_name=env_name, 
                             output_dir=self.output_dir, config=self.config)

        else:
            raise ValueError(f"Unknown agent type: {self.config.agent}")
