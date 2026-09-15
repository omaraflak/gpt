import json

with open("hugo_qa.json", "r") as f:
    data = json.load(f)

text = "\n\n".join(
    f"<|user|>\n{qa["instruction"]}\n<|end|>\n<|assistant|>\n{qa["response"]}\n<|end|>"
    for qa in data
)

with open("hugo_qa.txt", "w") as f:
    f.write(text)
