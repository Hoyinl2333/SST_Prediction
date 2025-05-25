import os

envs = [
    "dataStructureProject", "first", "imgOperation", "kaiwu",
    "llm", "mathorcupPython", "pythonProject2", "test"
]

for env in envs:
    os.system(f"conda env remove -n {env} -y")
