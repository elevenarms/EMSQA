import json
import re
import html
import os
from bs4 import BeautifulSoup


def clean_text(text):
    # Unescape HTML, replace \/ with /, and remove extra spaces
    text = html.unescape(text).replace("\\/", "/")
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()

def parse_html(content):
    # Parse HTML
    soup = BeautifulSoup(content, "html.parser")

    # Extract JSON from the 'data-page' attribute
    app_div = soup.find("div", {"id": "app"})
    if not app_div:
        raise ValueError("No <div id='app'> found in the HTML.")

    json_data = app_div.get("data-page")
    if not json_data:
        raise ValueError("No JSON data found in 'data-page' attribute.")

    # Convert HTML escape codes to valid JSON
    json_data = json_data.replace("&quot;", '"')
    # print(json_data)
    # Use html.unescape to convert all HTML entities to their corresponding characters
    # json_data = html.unescape(json_data)

    # Parse JSON
    data = json.loads(json_data)

    # with open("./test1.json", "w") as file:
    #     json.dump(data, file, indent=4)

    # Extract the required question
    question_data = data.get("props", {}).get("question", {})
    question_text = clean_text(
        BeautifulSoup(question_data.get("content", ""), "html.parser").text
    )

    # Extract answer choices
    answers = question_data.get("answers", [])
    answer_choices = [
        clean_text(BeautifulSoup(ans["text"], "html.parser").text)
        for ans in answers
    ]

    # Find the correct answer
    # correct_answer_index = next((i for i, ans in enumerate(answers) if ans["is_correct"]), None)
    # correct_answer = chr(97 + correct_answer_index) if correct_answer_index is not None else "N/A"
    correct_indices = [i for i, ans in enumerate(answers) if ans["is_correct"]]
    correct_answers = [chr(97 + i) for i in correct_indices]


    # Extract explanation
    try:
        explanation = clean_text(question_data.get("answer_rationale", ""))
    except:
        explanation = None

    # # Print extracted data
    # print("Question:", question_text)
    # for i, choice in enumerate(answer_choices):
    #     print(f"{chr(97 + i)}. {choice}")
    # print("Answer:", correct_answer)
    # print("Explanation:", explanation)

    qa_dic = {
        "question": question_text,
        "choices": {chr(97 + i): choice for i, choice in enumerate(answer_choices)},
        "answer": correct_answers,
        "explanation": explanation
    }

    return qa_dic


def rex_extract(html_data):
    soup = BeautifulSoup(html_data, "html.parser")
    data_page = soup.find("div", id="app")["data-page"]
    decoded_data = html.unescape(data_page)

    # Extract the main quiz question block (exclude studentQuestionOfTheDay)
    question_block_match = re.search(r'"quizId":.*?"question":\s*\{(.*?)\}\s*,\s*"initStudentAnswer"', decoded_data, re.DOTALL)
    question_block = question_block_match.group(1) if question_block_match else None

    if not question_block:
        return None  # No quiz question found

    # Extract question text from "stripped_content"
    question_match = re.search(r'"stripped_content":\s*"([^"]+)"', question_block)
    question_text = clean_text(question_match.group(1)) if question_match else None

    # Extract answer rationale (explanation)
    rationale_match = re.search(r'"answer_rationale":\s*"([^"]*)"', question_block)
    explanation = clean_text(rationale_match.group(1)) if rationale_match else None

    # Extract answer items from the "answers" array
    answers_match = re.search(r'"answers":\s*\[(.*?)\]', question_block, re.DOTALL)
    choices = {}
    correct_letters = []

    if answers_match:
        answers_str = answers_match.group(1)
        # Match each answer item and its correctness
        answer_items = re.findall(
            r'\{"id":\s*\d+,\s*"text":\s*"([^"]+)",\s*"is_correct":\s*(\d)\s*\}', answers_str
        )
        for i, (text, is_correct) in enumerate(answer_items):
            letter = chr(97 + i)  # Convert index to letter ('a', 'b', etc.)
            cleaned_answer = clean_text(text)
            choices[letter] = cleaned_answer
            if is_correct == "1":
                correct_letters.append(letter)

    # Package the extracted QA pair
    extracted = {
        "question": question_text.strip() if question_text else None,
        "choices": choices,
        "answer": correct_letters,
        "explanation": explanation.strip() if explanation else None,
    }
    return extracted



if __name__ == "__main__":
    section = "Airway, Respiration, and Ventilation"
    # Load the MHTML file and extract the content
    html_files = f"../../log/app-emtprep-com/{section}"  # Replace with your actual file path


    for root, dirs, files in os.walk(html_files):
        for file in files:
            if file.endswith(".html"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    content = f.read()

                try:
                    qa_dic = parse_html(content)
                except:
                    qa_dic = rex_extract(content)
                
                qa_dic = rex_extract(content)

                idx = file.split("_raw")[0].strip()
                path = os.path.join(root, f"{idx}.json")
                with open(path, "w") as f:
                    json.dump(qa_dic, f, indent=4)