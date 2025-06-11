import os
from dotenv import load_dotenv
from typing import cast
import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

# Load environment variables from the .env file
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

# Raise error if API key is missing
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

@cl.on_chat_start
async def start():
    # Reference: https://ai.google.dev/gemini-api/docs/openai
    external_client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    model = OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=external_client
    )

    config = RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=True
    )

    # Initialize session state
    cl.user_session.set("chat_history", [])
    cl.user_session.set("config", config)

    # Agent with bold creator name in instructions
    agent: Agent = Agent(
        name="Assistant",
        instructions="""
You are a helpful and intelligent assistant created by **Saif Soomro**.
If a user asks who created you, proudly respond with "I was created by **Saif Soomro**."
Always be polite, informative, and concise in your responses.
""",
        model=model
    )
    cl.user_session.set("agent", agent)

    # Send welcome message with bold creator name
    await cl.Message(
        content="""
# 🤖 Panaversity AI Assistant  
**Created by Saif Soomro**  
Welcome! How can I help you today?
"""
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """Process incoming messages and generate responses."""
    # Send a temporary thinking message
    msg = cl.Message(content="Thinking...")
    await msg.send()

    # Retrieve agent and config from session
    agent: Agent = cast(Agent, cl.user_session.get("agent"))
    config: RunConfig = cast(RunConfig, cl.user_session.get("config"))

    # Get chat history
    history = cl.user_session.get("chat_history") or []
    history.append({"role": "user", "content": message.content})

    try:
        print("\n[CALLING_AGENT_WITH_CONTEXT]\n", history, "\n")
        
        result = Runner.run_sync(
            starting_agent=agent,
            input=history,
            run_config=config
        )

        response_content = result.final_output

        # Update the thinking message with actual response
        msg.content = response_content
        await msg.update()

        # Update session chat history
        cl.user_session.set("chat_history", result.to_input_list())

        # Log interaction
        print(f"User: {message.content}")
        print(f"Assistant: {response_content}")

    except Exception as e:
        msg.content = f"Error: {str(e)}"
        await msg.update()
        print(f"Error: {str(e)}")
