from google.adk.agents import Agent

root_agent =Agent(
    name="trial_agent",
    model="gemini-2.5-flash",
    instruction="You are a adivisor about current soil condition",
    description=" You are an agent that advices the soil state when a user asks "
)

