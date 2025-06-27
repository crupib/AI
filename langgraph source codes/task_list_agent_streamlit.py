import streamlit as st
st.set_page_config(page_title="📝 Task Manager", layout="centered")

from typing import Annotated, Sequence, TypedDict, List, Dict, Any
from dotenv import load_dotenv
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages

# Load environment variables (e.g., OpenAI API key)
load_dotenv()

# --------------------------------------------------------------------------- #
# Session State Setup
# --------------------------------------------------------------------------- #
if "tasks" not in st.session_state:
    st.session_state.tasks: List[Dict[str, Any]] = []

if "conversation_state" not in st.session_state:
    st.session_state.conversation_state: Dict[str, Any] = {"messages": []}

if "last_output" not in st.session_state:
    st.session_state.last_output: str = ""

if "download_data" not in st.session_state:
    st.session_state.download_data = None
    st.session_state.download_filename = "my_tasks.txt"

# --------------------------------------------------------------------------- #
# Task Functions and Tools
# --------------------------------------------------------------------------- #

def show_task_list() -> str:
    """Return the to-do list formatted as a string."""
    if not st.session_state.tasks:
        return "📋 No tasks yet."
    return "\n".join(
        f"{idx+1}. [{'✔' if t['done'] else ' '}] {t['description']}"
        for idx, t in enumerate(st.session_state.tasks)
    )

@tool
def add_task(description: str) -> str:
    """Add a task to the list."""
    st.session_state.tasks.append({"description": description, "done": False})
    return show_task_list()

@tool
def mark_done(index: int) -> str:
    """Mark a task (1-based index) as completed."""
    if 1 <= index <= len(st.session_state.tasks):
        st.session_state.tasks[index - 1]["done"] = True
        return show_task_list()
    return "❌ Invalid task number."

@tool
def save_tasks(filename: str) -> str:
    """Save tasks to a .txt file and enable download."""
    if not filename.endswith(".txt"):
        filename += ".txt"
    try:
        content = show_task_list()
        with open(filename, "w", encoding="utf-8") as fh:
            fh.write(content)
        # Store for download
        st.session_state.download_data = content
        st.session_state.download_filename = filename
        return f"✅ Tasks saved to '{filename}'. Ready to download."
    except Exception as exc:
        return f"❌ Failed to save tasks: {exc}"

TOOLS = [add_task, mark_done, save_tasks]
TOOL_MAP = {t.name: t for t in TOOLS}

# --------------------------------------------------------------------------- #
# LangChain Agent Setup
# --------------------------------------------------------------------------- #

class AgentState(TypedDict):
    """Conversation state for LangGraph-style tracking."""
    messages: Annotated[Sequence[BaseMessage], add_messages]

@st.cache_resource
def _make_model():
    return ChatOpenAI(model="gpt-4o").bind_tools(TOOLS)

MODEL = _make_model()

def task_agent(state: AgentState, user_input: str) -> AgentState:
    """Run one cycle of the assistant, including tool handling."""
    system_prompt = SystemMessage(content=(
        "You are a helpful personal task manager assistant.\n\n"
        "Use 'add_task' to add tasks.\n"
        "Use 'mark_done' to mark them done.\n"
        "Use 'save_tasks' to save them.\n\n"
        f"Current list:\n{show_task_list()}"
    ))

    user_msg = HumanMessage(content=user_input.strip())
    history = list(state["messages"])
    response = MODEL.invoke([system_prompt] + history + [user_msg])

    updated_messages = history + [user_msg, response]

    # Handle tool calls
    if getattr(response, "tool_calls", None):
        for call in response.tool_calls:
            tool_name: str = call["name"]
            tool_args: Dict[str, Any] = call.get("args", {})
            tool_fn = TOOL_MAP.get(tool_name)

            if tool_fn is None:
                tool_output = f"❌ Unknown tool: {tool_name}"
            else:
                try:
                    tool_output = tool_fn.invoke(tool_args)
                except Exception as exc:
                    tool_output = f"❌ Tool error: {exc}"

            updated_messages.append(
                ToolMessage(tool_call_id=call["id"], content=tool_output)
            )

    return {"messages": updated_messages}

# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #

st.title("📝 LangChain Task Manager")

user_input = st.text_input(
    "Enter a task, mark done (e.g., 'mark 2'), or save (e.g., 'save tasks.txt'):"
)

if st.button("Submit") and user_input.strip():
    st.session_state.conversation_state = task_agent(
        st.session_state.conversation_state,
        user_input,
    )

    for msg in reversed(st.session_state.conversation_state["messages"]):
        if isinstance(msg, ToolMessage):
            st.session_state.last_output = msg.content
            break

if st.session_state.last_output:
    st.text_area("📋 Task List", value=st.session_state.last_output, height=200)

st.divider()

if st.button("Reset All"):
    st.session_state.tasks = []
    st.session_state.conversation_state = {"messages": []}
    st.session_state.last_output = ""
    st.session_state.download_data = None
    st.session_state.download_filename = "my_tasks.txt"
    st.success("✅ All tasks and state cleared.")

st.divider()

# Auto-show download button after save_tasks
if st.session_state.download_data:
    st.download_button(
        label=f"📥 Click to download: {st.session_state.download_filename}",
        data=st.session_state.download_data,
        file_name=st.session_state.download_filename,
        mime="text/plain",
        key="auto_download",
    )
