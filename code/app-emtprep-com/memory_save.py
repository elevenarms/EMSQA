import json
import os

# section = "Airway, Respiration, and Ventilation"
section = "Cardiology and Resuscitation"

memory = {}
for root, dirs, files in os.walk(f"../../log/app-emtprep-com/{section}"):
    for file in files:
        if file.endswith(".json"):
            # print(os.path.join(root, file))
            with open(os.path.join(root, file), "r") as f:
                data = json.load(f)
            
            q = data["question"].strip().lower()

            if q not in memory:
                memory[q] = {
                    "choices": data["choices"],
                    "answer": data["answer"],
                    "explanation": data["explanation"],
                }

print(f"Number of collected questions: {len(memory)}")

with open(f"../../log/app-emtprep-com/{section}/memory.json", "w") as f:
    json.dump(memory, f, indent=4)