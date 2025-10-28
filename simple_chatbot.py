import os
from vertexai.preview.agent import Agent, Tool
from vertexai.preview.language_models import ChatModel

# -----------------------------
# Step 1: Set up environment
# -----------------------------
# Make sure you have set:
# export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/sa-key.json"

# -----------------------------
# Step 2: Define a simple tool
# -----------------------------
def math_echo_tool(query: str):
    """Example tool that echoes math queries"""
    return f"[Math Tool Echo] You asked: {query}"

tool = Tool(
    name="MathEcho",
    description="Echoes math-related queries",
    func=math_echo_tool
)

# -----------------------------
# Step 3: Define the agent
# -----------------------------
agent = Agent(
    name="SimpleChatAgent",
    description="A minimal chatbot agent with one tool",
    tools=[tool]
)

# -----------------------------
# Step 4: Add LLM chat model
# -----------------------------
chat_model = ChatModel.from_pretrained("chat-bison@001")
chat = chat_model.start_chat()

# -----------------------------
# Step 5: Interactive CLI loop
# -----------------------------
print("=== Simple ADK Chatbot ===")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Run the agent on input
    response = agent.run(user_input)
    
    # If you want the response also to use the LLM chat directly:
    llm_response = chat.send_message(user_input)
    
    print("Agent (tools+LLM):", response)
    print("Agent (LLM only):", llm_response.text)
    print("-" * 40)
