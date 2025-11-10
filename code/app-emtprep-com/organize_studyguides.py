import os
import json
import re
from pdfminer.high_level import extract_text
from tqdm import tqdm
from transformers import MllamaForConditionalGeneration, AutoProcessor
import transformers
import torch

# model_name_or_path = "m42-health/Llama3-Med42-70B"
# model_name_or_path = "ProbeMedicalYonseiMAILab/medllama3-v20"
# model_name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_name_or_path = "meta-llama/Meta-Llama-3-70B-Instruct"
# model_name_or_path = "meta-llama/Meta-Llama-3.1-70B-Instruct"
# model_name_or_path = "meta-llama/Meta-Llama-3.1-405B-Instruct"
# model_name_or_path = "meta-llama/Meta-Llama-3.1-405B-Instruct-FP8"
# model_name_or_path = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
# model_name_or_path = "meta-llama/Llama-3.3-70B-Instruct"

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_name_or_path,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )

def apply_medllama3(messages, temperature=0.7, max_tokens=-1, top_k=150, top_p=0.75):
    prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if max_tokens != -1:
        outputs = pipeline(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        response = outputs[0]["generated_text"][len(prompt) :]
    else:
        outputs = pipeline(
            prompt,
            max_new_tokens=8192,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        response = outputs[0]["generated_text"][len(prompt) :]
    return response

def extract_json(response, pattern = r'\{.*\}'):
    # Regular expression pattern to match JSON content

    # Search for the pattern in the text
    # match = re.search(pattern, response, re.DOTALL)
    matches = re.findall(pattern, response, re.DOTALL)

    if not matches:
        print("No JSON object found in the text.")
        print(response)
        # print(json_data)
        return None, None

    json_data = matches[0] if len(matches) == 1 else matches[-1]
    
    try:
        # Load the JSON data
        data = json.loads(json_data)
        return None, data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        # print(response)
        # print(json_data)
        return e, json_data

def handleError(messages, next_response):
    error, next_response_dict = extract_json(next_response)
    print(error)
    print(next_response)
    ################################ no json file, regenerate ################################
    cnt = 1
    while error == None and next_response_dict == None and next_response:
        print(f"No json, repeat generating for the {cnt} time")
        raw_prompt = messages[0]["content"]
        prompt = "Plseas return the result in the defined json. " + raw_prompt
        messages[0]["content"] = prompt
        next_response = apply_medllama3(messages, temperature=0.7)
        error, next_response_dict = extract_json(next_response)
        cnt += 1

    ################################ json file incorrect ################################
    cnt = 1
    while error and cnt < 10:
        print(f"fix error for the {cnt} time")
        prompt = f"""There is an Error decoding JSON: {error} in the following json data
        {next_response_dict}, Can you fix the error and return the correct json format. Directly return the json without explanation.
        """
        messages = [{"role": "user", "content": prompt}]
        new_response = apply_medllama3(messages, temperature=0.3)
        print(new_response)
        error, next_response_dict = extract_json(new_response)
        cnt += 1
    
    if error:
        prompt = f"""There is an Error decoding JSON: {error} in the following json data
        {next_response_dict}, Can you fix the error and return the correct json format. Make sure it can be loaded using python (json.loads()). Directly return the json without explanation.
        """
        messages = [{"role": "user", "content": prompt}]
        new_response = apply_medllama3(messages, temperature=0.3)
        next_response_dict = json.loads(new_response)
    return next_response_dict

def organize_knowledge(sec, raw_text):
    prompt = f"""
    The following text has been extracted from a textbook section about "{sec}". 
    Your task is to format it properly while preserving all content exactly as it is.
    
    Instructions:
    1. Organize the text into properly formatted paragraphs.
    2. Identify and structure subtitles correctly (e.g., "Introduction", "Symptoms", etc.).
    3. If bullet points or lists are detected, format them accordingly.
    4. Keep all content **unaltered** (no summarization, no additional content).
    5. Return the result in structured JSON format: {{"subtitle": "paragraph", ...}}

    Extracted Text:
    ```
    {raw_text}
    ```

    Well-organized JSON Output:
    """

    messages = [{"role": "user", "content": prompt}]
    response = apply_medllama3(messages, temperature=0.7)
    # print(response)
    error, jsonfile = extract_json(response, pattern = r'\{.*\}')
    
    if error:
        jsonfile = handleError(messages, response)
        if not jsonfile:
            raise Exception("after handling error, there is still no json file")
    
    if not jsonfile:
        raise Exception("json file empty")

    return jsonfile

def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)

def convert_filename(pdf_filename):
    """
    Converts a PDF filename like 'abdominal-trauma-2024-08-13-213540.pdf'
    to a text filename like 'abdominal-trauma.txt'.
    """
    # Remove the date and timestamp pattern (YYYY-MM-DD-HHMMSS)
    cleaned_name = re.sub(r'-\d{4}-\d{2}-\d{2}-\d{6}', '', pdf_filename)

    # Replace .pdf with .txt
    txt_filename = os.path.splitext(cleaned_name)[0]
    return txt_filename

def pdf2text():
    root = "./study guides"
    levels = ["EMR", "EMT", "AEMT", "Paramedic"]
    for level in levels:

        if not os.path.exists(f"../../log/app-emtprep-com/study guides/{level}"):
            os.makedirs(f"../../log/app-emtprep-com/study guides/{level}")

        for file in os.listdir(os.path.join(root, level)):
            pdf_path = os.path.join(root, level, file)
            print(f"working with {pdf_path}..")
            pdf_name = convert_filename(file)
            text = extract_text_from_pdf(pdf_path)

            with open(f"../../log/app-emtprep-com/study guides/{level}/{pdf_name}.txt", "w", encoding="utf-8") as f:
                f.write(text)

def organize_level_knowledge():
    levels = ["EMR", "EMT", "AEMT", "Paramedic"]
    root = "../../log/app-emtprep-com/study guides"

    for level in levels:
        path = os.path.join(root, level)
        for file in os.listdir(path):

            # if file.endswith(".json"):
            #     with open(os.path.join(path, file), "r") as f:
            #         jsonfile = json.load(f)
            #     if jsonfile:
            #         continue
            #     else:
            #         print(f"{os.path.join(path, file)} is found empty, regenerate")

            if file.endswith(".txt"):
                print(f"working with {os.path.join(path, file)}...")
                with open(os.path.join(path, file), "r") as f:
                    content = f.read()
                
                section_title = file.replace(".txt", "")

                if f"{section_title}.json" in os.listdir(path):
                    with open(os.path.join(path, f"{section_title}.json"), "r") as f:
                        jsonfile = json.load(f)
                    if jsonfile:
                        continue
                    else:
                        print(f"{os.path.join(path, file)} is found empty, regenerate")


                jsonfile = organize_knowledge(sec=section_title, raw_text=content)
                with open(os.path.join(path, f"{section_title}.json"), "w") as f:
                    json.dump(jsonfile, f, indent=4)


def combine_knowledge():
    path = "../../log/app-emtprep-com/study guides"
    save_path = "../../knowledge/app-emtprep-com/study guides"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    levels = ["EMR", "EMT", "AEMT", "Paramedic"]
    study_guides = {}
    for level in levels:
        level_knowledge = {}
        cnt = 0
        for file in os.listdir(os.path.join(path, level)):
            if file.endswith(".json"):
                cnt += 1
                with open(os.path.join(path, level, file), "r") as f:
                    content = json.load(f)
                file_name = file.replace(".json", "")

                level_knowledge[file_name] = content

                if file_name not in study_guides:
                    study_guides[file_name] = {}
                for k, v in content.items():
                    
                    if k == "Resources" or k == "References":
                        continue

                    if k.lower() not in study_guides[file_name]:
                        if isinstance(v, list):
                            print(level, file)
                            v_text = "\n".join(v)
                            study_guides[file_name][k.lower()] = v_text.lower()
                        
                        elif isinstance(v, dict):
                            print(level, file)
                            text_lines = []
                            for k_, v_ in v.items():
                                text_lines.append(f"{k_}: {v_}")
                            v_text = "\n".join(text_lines)
                            study_guides[file_name][k.lower()] = v_text.lower()

                        else:
                            study_guides[file_name][k.lower()] = v.lower()

        with open(os.path.join(save_path, f"{level}.json"), "w") as f:
            json.dump(level_knowledge, f, indent=4)
        print(f"#study_guides in {level}: {len(level_knowledge)}")

    with open("../../knowledge/app-emtprep-com/study guides/combine.json", "w") as f:
        json.dump(study_guides, f, indent=4)
    print(f"#study_guides in total: {len(study_guides)}")

if __name__ == "__main__":
    # save knowledge to log
    # organize_level_knowledge()

    # save level knowledge
    combine_knowledge()