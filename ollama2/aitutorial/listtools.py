import llama_index.core.tools as tools

# List all attributes in FunctionTool
function_tool_attrs = dir(tools.FunctionTool)

# Filter out the methods/functions
function_tool_methods = [attr for attr in function_tool_attrs if callable(getattr(tools.FunctionTool, attr))]

print(function_tool_methods)
