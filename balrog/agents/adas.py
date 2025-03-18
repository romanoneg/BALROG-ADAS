import re
import json
import inspect
from balrog.agents.base import BaseAgent
from balrog.prompt_builder.history import Message
import adas_prompts

class adasAgent(BaseAgent):

    """Agent following the Automatic Design of Agentic Systems outlined in https://arxiv.org/pdf/2408.08435"""
    def __init__(self, client_factory, prompt_builder):
        super.__init__(client_factory, prompt_builder)
        self.client = client_factory() # nothing changes about setting up the agent
        # TODO: create init agents
        self.agent = None # turn this into an archive of agents 
        self.sys_prompt = Message(role="system", 
                content="You are a helpful assistant. Make sure to return in a WELL-FORMED JSON object")
        self.full_prompt = None
        self.init_obs = None

    def generate_agent(self):
        
        full_prompt = Message(role="user", attachement=None, content=
                              adas_prompts.prompt_instruction 
                              + "\n" + self.prompt_builder.system_prompt # not to be confused w/ self.sys_prompt
                              + "\n\n" + adas_prompts.framework_instructions
                              + "\n\n" + adas_prompts.example_agents
                              + "\n\n" + adas_prompts.task_instruction
                              + "\n\n" + self.init_obs
                              + "\n\n" + adas_prompts.output_instruction
                              + "\n\n" + adas_prompts.wrong_impl)

        self.full_prompt = full_prompt
        response = self.client.generate([self.sys_prompt, full_prompt])

        return _extract_json(response.completion)

    def errorReflect(self, cur_agent, error):
        prompt_instruction = lambda error: f"""
        Error during evaluation:
        {error}
        Carefully consider where you went wrong in your latest implementation. Using insights from previous
        attempts, try to debug the current code to implement the same thought. Repeat your previous thought in
        “thought”, and put your thinking for debugging in “debug_thought”.
        """.strip()

        full_prompt = Message(role="user",
                              content= cur_agent + "\n" + prompt_instruction(error))
        new_agent = self.client.generate([self.sys_prompt, full_prompt])

        return json.load(new_agent)

    def selfReflect(self, agent):

        reflect prompt = Message(role="user",
                    content= agent + "\n" + adas_prompts.reflect_instructions)

        feedback = self.client.generate([self.sys_prompt, self.full_prompt, full_prompt])
        feedback_message = Message(role="assistant", content=feedback.completion)

        full_reflection_prompt = Message(role="user", attachement=None, 
                                         content= adas_prompts.second_reflection)

        final_feedback = self.client.generate([self.sys_prompt, self.full_prompt, reflect_prompt, 
                                               feedback_message, full_reflection_prompt])
        
        return _extract_json(final_feedback.completion)

    def act(self, obs, prev_action=None):

        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)

        obs_input = self.prompt_builder.get_prompt()

        # First iteration generate an agent
        if self.agent is None:
            self.init_obs = obs_input
            # generate the new agent
            new_agent = self.generate_agent(obs_input)

        # TODO: make this real
        if "Your previous output did not contain a valid action." in obs['text']:            
            self.agents = self.errorReflect(self.agents, 
                            "Your previous output did not contain a valid action")

        if adas_iteration:
            # self reflect the agent
            self.agents = self.selfReflect(self.agents, obs_input)

        # MAKE THIS MORE RIGOROUS TOO
        forward_func = exec(self.agent['code'])

        return forward_func(obs_input)

# Util functions
def _extract_json(agent_guess):
    for _ in range(5):
        try:
            json_match = re.search(r'[{[](.+?)[}\]]', agent_guess, re.DOTALL)
            if not json_match:
                raise ValueError("Unable to find JSON, make sure output is valid JSON")
            agent = json.loads(agent_guess[json_match.start():json_match.end()])
            return agent
        except Exception as e:
            agent_guess = errorReflect(agent_guess, error=str(e)) 
    # TODO
    return ERROR?

# TODO check if function has right signature?
def _extract_agent_function(agent):
    try:
        if not agent_guess.strip()[:12] == "def forward(":
            raise ValueError("foward function not in valid format, make sure to output only code in the 'code' section.")
        # This is the part where we pray
        exec(agent_guess)
        return forward
    except Exception as e:
        agent_guess = errorReflect(self.agent, error=str(e)) 
    # TODO
    return ERROR?


