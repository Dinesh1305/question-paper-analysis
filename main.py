import os
import re
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import docx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ✅ Set path to your working Tesseract installation
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')


def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_text_from_pdf(file_path):
    """Extract text from digitally generated PDFs."""
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def extract_text_from_scanned_pdf(file_path):
    """Extract text using OCR from scanned PDFs."""
    text = ""
    with fitz.open(file_path) as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=300)
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            ocr_text = pytesseract.image_to_string(image)
            text += ocr_text + "\n"
    return text


def extract_questions(text):
    text = text.replace('\n', ' ')

    # Remove known section headers and instructional lines
    text = re.sub(r'section[- ]?[a-c]\b.*?(?=\d+\s*[\.\)])', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(this question|students.*?attempt|consist[s]? of|carries|each question|compulsory)\b.*?(?=\d+\s*[\.\)])', '', text, flags=re.IGNORECASE)

    # Split by numbered question format like "1." or "2)"
    raw_questions = re.split(r'(?:\d+\s*[\.\)]\s*)', text)

    questions = []
    for q in raw_questions:
        q = q.strip()

        # Clean common exam metadata
        q = re.sub(r'\(?\s*\d+\s*(marks|mark)?\s*\)?', '', q, flags=re.IGNORECASE)
        q = re.sub(r'\b(total|question[s]?|code|date|time|roll no|max:instructions|subject)\b.*?:.*?', '', q, flags=re.IGNORECASE)

        # Ignore known junk/instructional lines
        ignore_patterns = [
            r'^hours.*max:instructions$',
            r'^answer all questions.*$',
            r'^part\s?[a-c]\s?(and)?\s?part\s?[a-c]?.*$',
            r'^question\s*no\.?\s*is$',
            r'^question\s*no\.?\s*$',
        ]

        if any(re.match(pat, q.strip().lower()) for pat in ignore_patterns):
            continue

        # Filter out short or irrelevant lines
        if len(q) > 10 and "b.tech" not in q.lower() and "subject" not in q.lower():
            questions.append(q.strip())

    return questions



def load_questions():
    questions = []
    # ✅ Use the TestCase folder inside the project directory
    current_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "TestCase")
    current_folder = os.path.abspath(current_folder)

    for filename in os.listdir(current_folder):
        filepath = os.path.join(current_folder, filename)

        if filename.endswith('.txt'):
            content = extract_text_from_txt(filepath)
        elif filename.endswith('.docx'):
            content = extract_text_from_docx(filepath)
        elif filename.endswith('.pdf'):
            content = extract_text_from_pdf(filepath)
            if not content.strip():  # If no text found, try OCR
                print(f"🔍 Trying OCR on scanned PDF: {filename}")
                content = extract_text_from_scanned_pdf(filepath)
        else:
            continue

        extracted = extract_questions(content)
        questions.extend(extracted)

    cleaned_questions = [q.lower().strip() for q in questions if len(q.strip()) > 10]
    return cleaned_questions


def group_similar_questions(questions, similarity_threshold=0.75):
    embeddings = model.encode(questions)
    sim_matrix = cosine_similarity(embeddings)

    visited = set()
    groups = []

    for i in range(len(questions)):
        if i in visited:
            continue
        group = [questions[i]]
        visited.add(i)
        for j in range(i + 1, len(questions)):
            if j not in visited and sim_matrix[i][j] >= similarity_threshold:
                group.append(questions[j])
                visited.add(j)
        groups.append(group)
    return groups


def display_question_groups(groups):
    print("\n📊 Similar Question Groups with Frequencies:\n")
    for i, group in enumerate(groups, 1):
        print(f"\nGroup {i} (Count: {len(group)}):\n")
        for question in group:
            print(f"- {question}")


def display_unmatched(groups, all_questions):
    grouped_questions = {q for group in groups for q in group}
    unmatched = [q for q in all_questions if q not in grouped_questions]

    if unmatched:
        print("\n❗ Unmatched Questions:")
        for q in unmatched:
            print("-", q)


if __name__ == "__main__":


    all_questions = load_questions()
    similar_groups = group_similar_questions(all_questions, similarity_threshold=0.75)
    display_question_groups(similar_groups)
    display_unmatched(similar_groups, all_questions)

   print("")