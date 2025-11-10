import json
import os
import re


def combine_all_qa():
    q_set = []
    all_qa = []
    path = "../../log/app-emtprep-com/quiz"
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".json") and file != "memory.json":
                with open(os.path.join(root, file), "r") as f:
                    data = json.load(f)
                q = data["question"]

                if q not in q_set:
                    q_set.append(q)
                    all_qa.append(data)
    with open("../../data/app-emtprep-com/combine.json", "w") as f:
        json.dump(all_qa, f, indent=4)
    print(f"total number of qa in app-emtprep-com: {len(all_qa)}")


def combine_category():

    airway_respiration_and_ventilation = {
        "EMR": ["Airway, Respiration, and Ventilation"],
        "EMT": ["Airway, Respiration, and Ventilation"],
        "AEMT": ["Airway, Respiration, and Ventilation"],
        "Paramedic": ["Airway, Respiration, and Ventilation"],
        "Critical Care": ["CC-Advanced Airway Management"],
    }
    cardiology_and_resuscitation = {
        "EMR": ["Cardiology and Resuscitation"],
        "EMT": ["Cardiology and Resuscitation"],
        "AEMT": ["Cardiology and Resuscitation"],
        "Paramedic": ["Cardiology and Resuscitation"],
        "Critical Care": ["CC-Cardiac Emergencies"]
    }
    ems_operations = {
        "EMR": ["EMS Operations"],
        "EMT": ["EMS Operations"],
        "AEMT": ["EMS Operations"],
        "Paramedic": ["EMS Operations"],
        "Critical Care": ["CC-Operations"]
    }
    medical_and_obstetrics_gynecology = {
        "EMR": ["Medical, OB/Gyn"],
        "EMT": ["Medical, OB-Gyn"],
        "AEMT": ["Medical, OB/Gyn"],
        "Paramedic": ["Medical, OB/Gyn"],
        "Critical Care": ["CC-Medical Emergencies", "CC-OB & Pediatrics"]
    }
    trauma = {
        "EMR": ["Trauma"],
        "EMT": ["Trauma"],
        "AEMT": ["Trauma"],
        "Paramedic": ["Trauma"],
        "Critical Care": ["CC-Trauma and Burns"]
    }

    other = {
        "Critical Care": ["CC-Lab Values & Diagnostic Testing"]
    }

    all_category = {
        "airway_respiration_and_ventilation": airway_respiration_and_ventilation,
        "cardiology_and_resuscitation": cardiology_and_resuscitation,
        "ems_operations": ems_operations,
        "medical_and_obstetrics_gynecology": medical_and_obstetrics_gynecology,
        "trauma": trauma,
        "other": other
    }

    root = "../../log/app-emtprep-com/quiz"
    for k, vdct in all_category.items():
        category_qa = []
        q_set = []
        for k_, vlst in vdct.items():
            for l in vlst:
                path = os.path.join(root, k_, l)
                for root_, dirs, files in os.walk(path):
                    if "memory.json" in files:
                        for dir in dirs:
                            for file in os.listdir(os.path.join(root_, dir)):
                                if file.endswith(".json"):
                                    with open(os.path.join(root_, dir, file), "r") as f:
                                        cur = json.load(f)
                                    q = cur["question"]
                                    if q not in q_set:
                                        q_set.append(q)
                                        category_qa.append(cur)

        with open(f"../../data/app-emtprep-com/{k}.json", "w") as f:
            json.dump(category_qa, f, indent=4)
        print(f"#QAs {k}: {len(category_qa)}")



    # print(f"LEVEL: {level}")
    # for root, dirs, files in os.walk(f"../../data/app-emtprep-com/quiz/{level}"):
    #     if "memory.json" in files:
    #         category_qa = []
    #         q_set = []
    #         for dir in dirs:
    #             for file in os.listdir(os.path.join(root, dir)):
    #                 if file.endswith(".json"):
    #                     with open(os.path.join(root, dir, file), "r") as f:
    #                         cur = json.load(f)
    #                     q = cur["question"]
    #                     if q not in q_set:
    #                         q_set.append(q)
    #                         category_qa.append(cur)

    #         with open(f"{root}/qa.json", "w") as f:
    #             json.dump(category_qa, f, indent=4)
    #         print(f"number of qa in {os.path.basename(root)}: {len(category_qa)}")


def combine_level():
    levels = ["EMR", "EMT", "AEMT", "Paramedic", "Critical Care"]
    for level in levels:
        print(f"LEVEL: {level}")
        category_qa = []
        q_set = []
        for root, dirs, files in os.walk(f"../../log/app-emtprep-com/quiz/{level}"):
            if "memory.json" in files:
                for dir in dirs:
                    for file in os.listdir(os.path.join(root, dir)):
                        if file.endswith(".json"):
                            with open(os.path.join(root, dir, file), "r") as f:
                                cur = json.load(f)
                            q = cur["question"]
                            if q not in q_set:
                                q_set.append(q)
                                category_qa.append(cur)

        with open(f"../../data/app-emtprep-com/{level.lower()}.json", "w") as f:
            json.dump(category_qa, f, indent=4)
        print(f"number of qa in {level}: {len(category_qa)}")


if __name__ == "__main__":

    combine_all_qa()

    combine_category()

    combine_level()