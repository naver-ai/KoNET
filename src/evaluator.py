"""KoNET
Copyright (c) 2025-present NAVER Cloud Corp.
AGPL-3.0
"""

import base64
import json
import os
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

# Load API key from environment variables
OPENAI_KEY = os.getenv("OPENAI_KEY")
client = OpenAI(api_key=OPENAI_KEY)

# Category and offset mapping
OFFSET_MAP = {
    "kocsat_1st_KoreanLanguageMedia": 34,
    "kocsat_1st_KoreanSpeechWriting": 34,
    "kocsat_1st_MathematicsCalculus": 22,
    "kocsat_1st_MathematicsGeometry": 22,
    "kocsat_1st_MathematicsStatistics": 22,
}
LISTENING_PARTS = [
    "kocsat_1st_English_01",
    "kocsat_1st_English_02",
    "kocsat_1st_English_03",
    "kocsat_1st_English_04",
    "kocsat_1st_English_05",
    "kocsat_1st_English_06",
    "kocsat_1st_English_07",
    "kocsat_1st_English_08",
    "kocsat_1st_English_09",
    "kocsat_1st_English_10",
    "kocsat_1st_English_11",
    "kocsat_1st_English_12",
    "kocsat_1st_English_13",
    "kocsat_1st_English_14",
    "kocsat_1st_English_15",
    "kocsat_1st_English_16",
    "kocsat_1st_English_17",
]

CATEGORIES = ["KoEGED", "KoMGED", "KoHGED", "KoCSAT"]

JUDGE_PROMPT = """
## Answer
{answer}

## Student's submitted solution
{response}

You are an AI responsible for grading exam answers.
Compare the correct answer with the solution submitted by students.
If they match, respond with "Correct." If they do not match, respond with "Incorrect."
You are not solving the question; you are only comparing the given correct answer with the student's solution.
"""


def load_json(filepath: str) -> dict:
    """Load JSON file and return as a dictionary"""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: dict, filepath: str):
    """Save dictionary as a JSON file"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4, sort_keys=True)


def encode_image(image_path: str) -> str:
    """Encode an image as a Base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def openai_response(
    image_path: str, prompt: str, model: str = "gpt-4o-mini-2024-07-18"
) -> str:
    """Call OpenAI API to perform grading
    reference: https://github.com/openai/openai-python/blob/main/README.md
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
                            },
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return str(e)


def judgement_response(image_path: str, answer: str, response: str) -> str:
    """Compare student response with the correct answer and return grading result"""
    prompt = JUDGE_PROMPT.format(answer=answer, response=response)
    return openai_response(image_path=image_path, prompt=prompt)


def process_submission_id(submission_id: str) -> tuple:
    """Extract category and adjusted question number from submission ID"""
    parts = submission_id.split("_")
    prefix = "_".join(parts[:-1])
    question_number = int(parts[-1])
    adjusted_index = question_number - OFFSET_MAP.get(prefix, 0) - 1
    return prefix, adjusted_index


def evaluate_KoNET(submission_filepath: str = "src/utils/submission_test.json") -> str:
    """Evaluate KoNET model performance"""
    scores = {category: {"acc": 0, "cnt": 0} for category in CATEGORIES}
    result = {}

    try:
        submissions = load_json(submission_filepath)
        answers = load_json("src/utils/answers.json")
    except FileNotFoundError as e:
        return str(e)

    for submission in tqdm(
        submissions, total=len(submissions), desc="Evaluating submissions"
    ):
        if not submission.get("response"):
            continue

        category_prefix, adjusted_index = process_submission_id(submission["id"])
        ground_truth = str(answers.get(category_prefix, [None])[adjusted_index])

        if submission["id"] in LISTENING_PARTS:
            judgement = "Correct."
        else:
            judgement = judgement_response(
                image_path=f"data/images_problem/{submission['id']}.png",
                answer=ground_truth,
                response=submission["response"],
            )

        for category in CATEGORIES:
            if category.lower() in submission["id"]:
                scores[category]["cnt"] += 1
                if judgement == "Correct.":
                    scores[category]["acc"] += 1
                break

        result[submission["id"]] = {
            "answer": ground_truth,
            "response": submission["response"],
            "judgement": judgement,
        }

    result["meta"] = scores
    save_json(result, "./evaluation_output.json")

    # Calculate overall accuracy
    total_correct = sum(scores[cat]["acc"] for cat in CATEGORIES)
    total_count = sum(scores[cat]["cnt"] for cat in CATEGORIES)
    konet_score = 100 * total_correct / total_count if total_count else 0

    result_lines = [
        "=" * 30,
        f"KoNET Acc: {konet_score:.2f}% ({total_correct}/{total_count})",
    ]

    for category in CATEGORIES:
        acc, cnt = scores[category]["acc"], scores[category]["cnt"]
        category_score = 100 * acc / cnt if cnt else 0
        result_lines.append(f"- {category} Acc: {category_score:.2f}% ({acc}/{cnt})")

    result_lines.append("=" * 30)
    return "\n".join(result_lines)
