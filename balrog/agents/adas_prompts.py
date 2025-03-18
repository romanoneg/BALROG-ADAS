framework_instructions = """
    # The following functions have been create to show examples of how you would 
    # query a model, these functions are not available in the code and are here
    # to demonstrate possible functions to write.

    def create_message(prompt):
        \"""
        The Message class has been created to make querying the model easier.

        Args: 
        - prompt (str): the current prompt to be passed to the model

        Returns:
        - msg (Message): the output Message object ready to be used to query the model

        The arguments of the Message class are:
            - role (str): From the list ["system", "user", "assistant"]
            - content (str): String content of the prompt
            - attachement (None): To be ignored from now and is completely optional
        \"""

        msg = Message(role="user", content=prompt)
        return msg
        

    def get_response_from_model(full_prompt):
        \"""
        Function for querying the model.

        Args:
        - full_prompt (Message): the Message object to be passed to the model

        Returns:
        - output (ResponseObject): the response as a string, can access the string
                        response with output.completion .

        Additionally, self.sys_prompt is:
        "You are a helpful assistant. Make sure to return in a WELL-FORMED JSON object",
        to parse the JSON objects use json.load() on the output.completion .
        \"""

        # self.client.generate expects a list of Message objects 
        output = self.client.generate([self.sys_prompt, prompt_message])
        
        return output
    """.strip()
            
            # Pass in observations manually at each step
            # def get_observations():
            #     \"""
            #     Function that returns an already formed list of Message objects with the current
            #     observations and past observations and actions. 
            #     \"""
            #
            #     observations = self.prompt_builder.get_prompt()
            #     return observations

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

    # example observation
    An example observation is given below:
    """.strip()

output_instruction = """
    # Output Instruction and Example:
    The first key should be (“thought”), and it should capture your thought process for designing the
    next function. In the “thought” section, first reason about what the next interesting agent to try
    should be, then describe your reasoning and the overall concept behind the agent design, and
    finally detail the implementation steps. The second key (“name”) corresponds to the name of
    your next agent architecture. Finally, the last key (“code”) corresponds to the exact “forward()”
    function in Python code that you would like to try. You must write COMPLETE CODE in “code”:
    Your code will be part of the entire project, so please implement complete, reliable, reusable code snippets.
    Here is an example of the output format for the next agent:
    {“thought”: “**Insights:** Your insights on what should be the next interesting agent. **Overall Idea:**
    your reasoning and the overall concept behind the agent design. **Implementation:** describe the
    implementation step by step.”,
    “name”: “Name of your proposed agent”,
    “code”: “def forward(self, taskInfo): # Your code here”}
    """.strip()

# TODO LIST OF WRONG
wrong_impl = """
    ## WRONG Implementation examples:
    LIST OF STUFF """


reflect_instruction =  """
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

second_reflection = """
    Using the tips in “## WRONG Implementation examples” section, further revise the code.
    Your response should be organized as follows:
    Include your updated reflections in the “reflection”. Repeat the previous “thought” and “name”. Update
    the corrected version of the code in the “code” section
    """.strip()

# TODO example agents
example_agents = """
    # EXAMPLE agents are given:
""".strip()
