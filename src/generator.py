"""KoNET
Copyright (c) 2025-present NAVER Cloud Corp.
AGPL-3.0
"""

import io
import json
import os
import zipfile
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

# Define paths
DATA_DIR = Path("data")
FIGURES_DIR = DATA_DIR / "figures"
IMAGES_DIR = DATA_DIR / "images"
PROBLEM_IMAGES_DIR = DATA_DIR / "images_problem"
BBOX_FILE = Path("src/utils/bboxes.json")
OFFSET_MAP = {
    "kocsat_1st_KoreanLanguageMedia": 34,
    "kocsat_1st_KoreanSpeechWriting": 34,
    "kocsat_1st_MathematicsCalculus": 22,
    "kocsat_1st_MathematicsGeometry": 22,
    "kocsat_1st_MathematicsStatistics": 22,
}

# Set headers for HTTP requests
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36",
    "Referer": "https://www.kice.re.kr/",
}


def create_directory(path: Path) -> None:
    """Create a directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, filepath: Path) -> bool:
    """Download a file from a URL and save it locally."""
    try:
        response = requests.get(url, headers=HEADERS, stream=True)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        return True
    except requests.RequestException as e:
        print(f"Failed to download {url}: {e}")
        return False


def unzip_file(zip_filepath: Path, extract_to: Path, rename_mapping: dict) -> None:
    """Extract and rename files from a ZIP archive."""
    try:
        with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
            zip_ref.extractall(extract_to)

        for old_name, new_name in rename_mapping.items():
            old_path, new_path = extract_to / old_name, extract_to / new_name
            if old_path.exists():
                os.rename(old_path, new_path)
    except zipfile.BadZipFile as e:
        print(f"Invalid ZIP file {zip_filepath}: {e}")


def process_files(file_urls: dict, rename_mappings: dict) -> None:
    """Download and process files, extracting ZIP archives if necessary."""
    create_directory(FIGURES_DIR)
    for url, filename in tqdm(
        file_urls.items(),
        desc="[1/3] Downloading files",
        total=72,
    ):
        filepath = FIGURES_DIR / filename
        if not filepath.exists() and not download_file(url, filepath):
            continue
        if filename.endswith(".zip"):
            unzip_file(filepath, FIGURES_DIR, rename_mappings)


def extract_images_from_pdf(pdf_path: Path, zoom: int = 2) -> None:
    """Extract images from a PDF file and save them as PNG."""
    doc = fitz.open(pdf_path)
    create_directory(IMAGES_DIR)

    for page_idx, page in enumerate(doc):
        output_filename = IMAGES_DIR / f"{pdf_path.stem}_{page_idx}.png"
        if not output_filename.exists():
            image = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            image.save(output_filename)


def convert_pdf_to_images() -> None:
    """Convert all PDFs in the figures directory to images."""
    for pdf_file in tqdm(
        FIGURES_DIR.glob("*.pdf"),
        desc="[2/3] Converting PDFs to images",
        total=100,
    ):
        extract_images_from_pdf(pdf_file)


def load_json(file_path: Path):
    """Load a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def crop_images(img_dir: Path, img_paths: list, bboxes: list):
    """Crop images using bounding boxes and merge them if necessary."""
    cropped_images = [
        Image.open(img_dir / img).crop(bbox) for img, bbox in zip(img_paths, bboxes)
    ]
    if len(cropped_images) == 1:
        return cropped_images[0]

    width, height = max(img.width for img in cropped_images), sum(
        img.height for img in cropped_images
    )
    merged_img = Image.new("RGB", (width, height), (255, 255, 255))
    y_offset = 0
    for img in cropped_images:
        merged_img.paste(img, (0, y_offset))
        y_offset += img.height
    return merged_img


def extract_problem_images() -> None:
    """Extract and save problem images using bounding boxes."""
    create_directory(PROBLEM_IMAGES_DIR)
    bbox_data = load_json(BBOX_FILE)

    for doc in tqdm(
        bbox_data,
        desc="[3/3] Extracting problem images",
        total=2377,
    ):
        output_path = PROBLEM_IMAGES_DIR / f"{doc['idx']}.png"
        if not output_path.exists():
            cropped_img = crop_images(IMAGES_DIR, doc["img_path"], doc["bbox"])
            cropped_img.save(output_path)


def load_dataframe() -> pd.DataFrame:
    """Load and return a DataFrame with problem metadata."""
    types, points, errors, answers = map(
        load_json,
        [
            Path("src/utils/types.json"),
            Path("src/utils/points.json"),
            Path("src/utils/errors.json"),
            Path("src/utils/answers.json"),
        ],
    )
    data = []
    for img_file in PROBLEM_IMAGES_DIR.glob("*.png"):
        idx = img_file.stem
        idx_parts = idx.split("_")
        idx_prefix = "_".join(idx_parts[:-1])
        number = int(idx_parts[-1])

        if idx_prefix in OFFSET_MAP.keys():
            number = number - OFFSET_MAP[idx_prefix] - 1
        else:
            number = number - 1

        data.append(
            {
                "idx": idx,
                "img_path": img_file,
                "problem_type": (
                    str(types.get(idx_prefix)[number]) if idx_prefix in types else ""
                ),
                "problem_point": (
                    str(points.get(idx_prefix)[number]) if idx_prefix in points else ""
                ),
                "problem_answer": str(answers.get(idx_prefix)[number]),
                "problem_error": errors.get(idx, ""),
            }
        )
    return pd.DataFrame(data)


def generate_KoNET() -> None:
    """Main function to download, extract, and process exam data."""

    try:
        """
        We fully understand the copyright of the problems used in relation to KoNET and believe that all copyrights should be respected.
        Therefore, we discourage the direct inclusion of images or the provision of PDF files within KoNET.
        Instead, we provide guidelines that allow users to download PDF files through publicly accessible links and use them to generate benchmarks.

        We strongly recommend that this approach be used solely for research purposes, rather than for commercial use.
        Additionally, please note that the guidelines may be subject to change if any related issues arise in the future.

        If you have any questions, please feel free to contact us at any time.
        Thank you.
        """
        file_id = "1qOfYrkOzCzWvZxcWn60jLRTQZZuF48QN"
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        session = requests.Session()
        response = session.get(url, stream=True)
        file_content = io.BytesIO(response.content)
        FILE_URLS = json.load(file_content)
    except:
        """
        You can also download the file directly from the Korea Institute for Curriculum and Evaluation (KICE) at https://www.kice.re.kr/sub/info.do?m=0303&s=kice.
        After downloading, please save the file in the data/figures directory.
        Additionally, all file names must be written in English only. When renaming the files, please refer to the files under the src/utils directory.
        """
        FILE_URLS = {}

    RENAME_MAPPINGS = {
        "01 독일어Ⅰ_문제.pdf": "kocsat_1st_German1.pdf",
        "01 물리학Ⅰ_문제.pdf": "kocsat_1st_Physics1.pdf",
        "01 생활과 윤리_문제.pdf": "kocsat_1st_EthicsAndLife.pdf",
        "01 성공적인 직업생활_문제.pdf": "kocsat_1st_SuccessfulCareerLife.pdf",
        "02 농업 기초 기술_문제.pdf": "kocsat_1st_AgriculturalTechnology.pdf",
        "02 윤리와 사상_문제.pdf": "kocsat_1st_EthicsAndPhilosophy.pdf",
        "02 프랑스어Ⅰ_문제.pdf": "kocsat_1st_French1.pdf",
        "02 화학Ⅰ_문제.pdf": "kocsat_1st_Chemistry1.pdf",
        "03 공업 일반_문제.pdf": "kocsat_1st_IndustrialGeneral.pdf",
        "03 생명과학Ⅰ_문제.pdf": "kocsat_1st_Biology1.pdf",
        "03 스페인어Ⅰ_문제.pdf": "kocsat_1st_Spanish1.pdf",
        "03 한국지리_문제.pdf": "kocsat_1st_KoreanGeography.pdf",
        "04 상업 경제_문제.pdf": "kocsat_1st_BusinessEconomy.pdf",
        "04 세계지리_문제.pdf": "kocsat_1st_WorldGeography.pdf",
        "04 중국어Ⅰ_문제.pdf": "kocsat_1st_Chinese1.pdf",
        "04 지구과학Ⅰ_문제.pdf": "kocsat_1st_EarthScience1.pdf",
        "05 동아시아사_문제.pdf": "kocsat_1st_EastAsianHistory.pdf",
        "05 물리학Ⅱ_문제.pdf": "kocsat_1st_Physics2.pdf",
        "05 수산·해운 산업 기초_문제.pdf": "kocsat_1st_FisheryMaritimeIndustry.pdf",
        "05 일본어Ⅰ_문제.pdf": "kocsat_1st_Japanese1.pdf",
        "06 러시아어Ⅰ_문제.pdf": "kocsat_1st_Russian1.pdf",
        "06 세계사_문제.pdf": "kocsat_1st_WorldHistory.pdf",
        "06 인간 발달_문제.pdf": "kocsat_1st_HumanDevelopment.pdf",
        "06 화학Ⅱ_문제.pdf": "kocsat_1st_Chemistry2.pdf",
        "07 경제_문제.pdf": "kocsat_1st_Economics.pdf",
        "07 생명과학Ⅱ_문제.pdf": "kocsat_1st_Biology2.pdf",
        "07 아랍어Ⅰ_문제.pdf": "kocsat_1st_Arabic1.pdf",
        "08 베트남어Ⅰ_문제.pdf": "kocsat_1st_Vietnamese1.pdf",
        "08 정치와 법_문제.pdf": "kocsat_1st_PoliticsAndLaw.pdf",
        "08 지구과학Ⅱ_문제.pdf": "kocsat_1st_EarthScience2.pdf",
        "09 사회·문화_문제.pdf": "kocsat_1st_SocietyAndCulture.pdf",
        "09 한문Ⅰ_문제.pdf": "kocsat_1st_ClassicalChinese1.pdf",
    }
    process_files(FILE_URLS, RENAME_MAPPINGS)
    convert_pdf_to_images()
    extract_problem_images()
    return load_dataframe()


if __name__ == "__main__":
    generate_KoNET()
