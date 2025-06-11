framework_instructions = """
    # The following functions have been create to show examples of how you would 
    # query a model, these functions are not available in the code and are here
    # to demonstrate possible functions to write.

    # All messages to the model must be a Message object:
    # The allowed roles are ['system', 'user', 'assistant']:
    prompt_message = Message(role="user", content=this_is_my_prompt)

    # How to query the model:
    # a self.prompt_builder object is used to keep using the same context of observations
    # it has the following capabilities:

    # to add to the context the previous action taken by the model
    self.prompt_builder.update_action(prev_action)

    # to update the context with the current observation:
    self.prompt_builder.update_observation(obs)

    # to get a list of Message objects that represents the context of observations:
    messages = self.prompt_builder.get_prompt()
        
    # calling the model:
    # use self.client.generate which expects a list of Message objects 
    output = self.client.generate(messages + [prompt_message])
    # the output will be a LLMResponse object which looks like this:

    LLMResponse = namedtuple(
        "LLMResponse",
        [
            "model_id",
            "completion",
            "stop_reason",
            "input_tokens",
            "output_tokens",
            "reasoning",
        ],
    )

    # If you have to modify the output manually do so like this:
    output = output._replace(completion=replacement_completion)
    """.strip()

            

prompt_instruction = """
    You are an expert machine learning researcher testing various agentic systems. Your objective is to design
    building blocks such as prompts and control flows within these systems to solve complex tasks. Your aim
    is to design an optimal agent performing well on 
    """.strip()

task_instruction = """
    # Your task
    You are deeply familiar with prompting techniques and the agent works from the literature. Your goal is
    to maximize the specified performance metrics by proposing interestingly new agents.
    Observe the discovered agents carefully and think about what insights, lessons, or stepping stones can be
    learned from them.
    Be creative when thinking about the next interesting agent to try. You are encouraged to draw inspiration
    from related agent papers or academic papers from other research areas.
    Use the knowledge from the archive and inspiration from academic literature to propose the next
    interesting agentic system design.
    THINK OUTSIDE THE BOX.
    """.strip()

output_instruction = """
    # Output Instruction and Example:
    The first key should be (“thought”), and it should capture your thought process for designing the
    next function. In the “thought” section, first reason about what the next interesting agent to try
    should be, then describe your reasoning and the overall concept behind the agent design, and
    finally detail the implementation steps. The second key (“name”) corresponds to the name of
    your next agent architecture. Finally, the last key (“code”) corresponds to the exact “CustomAgent()”
    class in Python code that you would like to try. You must write COMPLETE CODE in “code”:
    Your code will be part of the entire project, so please implement complete, reliable, reusable code snippets.
    Here is an example of the output format for the next agent:
    {“thought”: “**Insights:** Your insights on what should be the next interesting agent. **Overall Idea:**
    your reasoning and the overall concept behind the agent design. **Implementation:** describe the
    implementation step by step.”,
    “name”: “Name of your proposed agent”,
    “code”: “class CustomAgent(BaseAgent):

                def __init__(self, client_factory, prompt_builder):
                    super().__init__(client_factory, prompt_builder)
                    self.client = client_factory()
                    # your code here
                    import ... # imports MUST be done inside of __init__
                
                def act(self, obs, prev_action=None) -> LLMResponse:
                    # your code here
                    # you may make multiple calls to the model here as long as you output a single LLMResponse object
    "}
    """.strip()

error_prompt = lambda x: f"""
    Error during evaluation:
    {x}
    Carefully consider where you went wrong in your latest implementation. Using insights from previous
    attempts, try to debug the current code to implement the same thought. Repeat your previous thought in
    “thought”, and put your thinking for debugging in “debug thought”.
    """

# TODO:
wrong_impl = """
    ## WRONG Implementation examples:
    - Not utilizing the prompt_builder to expand context.
    - Not making sure all output is in the form of a valid json.    
    - Importing outside Libraries -> you must only use base python.
    - DO NOT import libraries at the top of the block of code, add your imports INSIDE of the class!
    - Trying to add arguments to the class, the ONLY arguments passed to init are ONLY: client_factory and prompt_builder! NO CONFIG!
    - trying to import LLMResponse, it is already imported in the scope!
    - BY FAR the most common wrong implementation and the most important to get right is that your act method MUST output a LLMResponseobject that has ONLY the valid action in the "completion" field and ONLY the action!!
    - quotes within strings should be backslashed in order to keep correct string format.
    """.strip()


reflection_one =  """
    Carefully review the proposed new architecture and reflect on the following points:
    1. **Interestingness**: Assess whether your proposed architecture is interesting or innovative compared
    to existing methods in the archive. If you determine that the proposed architecture is not interesting,
    suggest a new architecture that addresses these shortcomings.
    - Make sure to check the difference between the proposed architecture and previous attempts.
    - Compare the proposal and the architectures in the archive CAREFULLY, including their actual differences
    in the implementation.
    - Decide whether the current architecture is innovative.
    - USE CRITICAL THINKING!
    2. **Implementation Mistakes**: Identify any mistakes you may have made in the implementation.
    Review the code carefully, debug any issues you find, and provide a corrected version. REMEMBER
    checking "## WRONG Implementation examples" in the prompt.
    3. **Improvement**: Based on the proposed architecture, suggest improvements in the detailed
    implementation that could increase its performance or effectiveness. In this step, focus on refining and
    optimizing the existing implementation without altering the overall design framework, except if you
    want to propose a different architecture if the current is not interesting.
    - Observe carefully about whether the implementation is actually doing what it is supposed to do.
    - Check if there is redundant code or unnecessary steps in the implementation. Replace them with
    effective implementation.
    - Try to avoid the implementation being too similar to the previous agent.
    And then, you need to improve or revise the implementation, or implement the new proposed architecture
    based on the reflection.
    Your response should be organized as follows:
    "reflection": Provide your thoughts on the interestingness of the architecture, identify any mistakes in the
    implementation, and suggest improvements.
    "thought": Revise your previous proposal or propose a new architecture if necessary, using the same
    format as the example response.
    "name": Provide a name for the revised or new architecture. (Don’t put words like "new" or "improved"
    in the name.)
    "code": Provide the corrected code or an improved implementation. Make sure you actually implement
    your fix and improvement in this code.
    """.strip()

reflection_two = """
    Using the tips in “## WRONG Implementation examples” section, further revise the code.
    Your response should be organized as follows:
    Include your updated reflections in the “reflection”. Repeat the previous “thought” and “name”. Update
    the corrected version of the code in the “code” section
    """.strip()

