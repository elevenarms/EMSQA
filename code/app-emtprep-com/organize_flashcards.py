import json
import os
from collections import defaultdict

if __name__ == "__main__":
    path = "../../log/app-emtprep-com/flashcards/"
    flashcards = defaultdict(dict)
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            category_flashcards = defaultdict(dict)
            for file in os.listdir(os.path.join(root, dir)):
                if file.endswith(".json"):
                    with open(os.path.join(root, dir, file), "r") as f:
                        cur = json.load(f)
                    topic = file.split(".json")[0]

                    category_flashcards[topic] = cur

                    if topic not in flashcards:
                        flashcards[topic] = {}
                    for k, v in cur.items():
                        if k.lower() not in flashcards[topic]:
                            flashcards[topic][k.lower()] = v.lower()
                        # else:
                        #     flashcards[topic][k.lower()] += " " + v.lower()
            with open("../../knowledge/app-emtprep-com/flashcards/{}.json".format(dir), "w") as f:
                json.dump(category_flashcards, f, indent=4)
            print(f"#flashcards in {dir}: {len(category_flashcards)}")

    with open("../../knowledge/app-emtprep-com/flashcards/combine.json", "w") as f:
        json.dump(flashcards, f, indent=4)
    print(f"#flashcards in total: {len(flashcards)}")



