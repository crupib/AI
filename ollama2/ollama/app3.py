import ollama
import tempfile
import os
import subprocess

modelfile_contents = """
FROM llama3
PARAMETER temperature 0.3
SYSTEM You are Peter, an intelligent assistant known for providing clear, concise, and informative answers to questions.
"""

# Create a temporary directory and Modelfile
with tempfile.TemporaryDirectory() as tmpdir:
    modelfile_path = os.path.join(tmpdir, "Modelfile")
    with open(modelfile_path, "w") as f:
        f.write(modelfile_contents)

    # Run the command and suppress all stdout/stderr
    subprocess.run(
        ["ollama", "create", "smartassistant", "--file", modelfile_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

# Now just generate a response — this will be the only printed output
res = ollama.generate(model="smartassistant", prompt="what is your name")
print(res["response"])

