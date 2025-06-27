import os
from typing import Sequence, Annotated, TypedDict
from operator import add as add_messages

from dotenv import load_dotenv
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage,
    BaseMessage,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langchain_chroma import Chroma
from langchain_core.tools import tool

load_dotenv()
base_llm = ChatOpenAI(model="gpt-4o", temperature=0)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

text_path = "hr_policy.txt"

if not os.path.exists(text_path):
    raise FileNotFoundError(f"Text file not found: {text_path}")

text_loader = TextLoader(text_path) 

documents = text_loader.load()
print(documents)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents) 

vector_store = Chroma.from_documents(chunks,embeddings)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4} # K is the amount of chunks to return
)

@tool
def policy_lookup(query: str) -> str:
    """Search the company policy handbook and return relevant sections."""
    docs = retriever.invoke(query)
    if not docs:
        return "No matching policy section found."
    out = []
    for i, d in enumerate(docs, 1):
        source = d.metadata.get("source", text_path)
        out.append(f"[{source} – chunk {i}]\n{d.page_content.strip()}")
    return "\n\n".join(out)        

tools = [policy_lookup]
llm = base_llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def _should_continue(state: AgentState) -> bool:
    """Check if the last message contains tool calls."""
    last = state["messages"][-1]
    return getattr(last, "tool_calls", None) not in (None, [])

system_prompt = """
    "You are the company's HR assistant. "
    "Answer employee questions using only the official policy handbook. "
    "Call the `policy_lookup` tool whenever you need to search. "
    "Always cite the chunk text you used in square brackets at the end of the sentences."
"""

tools_dict = {our_tool.name: our_tool for our_tool in tools} 

def _call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    msgs = list(state["messages"])
    if not msgs or not isinstance(msgs[0], SystemMessage):
        msgs.insert(0, SystemMessage(content=system_prompt))
    reply = llm.invoke(msgs)
    return {"messages": [reply]}

def _take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    last_msg = state["messages"][-1]
    tool_messages = []

    for call in last_msg.tool_calls:
        name = call["name"]
        args = call.get("args", {})
        query = args.get("query", "")
        if name not in tools_dict:
            content = "Invalid tool name."
        else:
            content = tools_dict[name].invoke(query)
        tool_messages.append(
            ToolMessage(tool_call_id=call["id"], name=name, content=content)
        )
    return {"messages": tool_messages}##

graph = StateGraph(AgentState)

graph.add_node("llm", _call_llm)
graph.add_node("tool_runner", _take_action)

graph.add_conditional_edges(
  "llm", 
  _should_continue, 
  {True: "tool_runner", False: END}
)

graph.add_edge("tool_runner", "llm")

graph.set_entry_point("llm")

hr_agent = graph.compile()

def _chat_cli() -> None: ##
    print("=== HR Assistant === (type 'quit' to exit)")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break

        messages = [HumanMessage(content=user_input)]
        
        result = hr_agent.invoke({"messages":messages})        
        print(result["messages"][-1].content)  

_chat_cli()


