from pathlib import Path
import pdfplumber 

def extract_text_from_pdf(pdf_folder: str, output_folder: str) -> int:
    """
    Extracts text from all PDF files in the specified folder and saves them as .txt files in the output folder.

    Args:
        pdf_folder (Path): Path to the folder containing PDF files.
        output_folder (Path): Path to the folder where extracted text files will be saved.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    n = 0
    for pdf_file in Path(pdf_folder).glob("*.pdf"):
        with pdfplumber.open(pdf_file) as doc:
            text = "\n".join([p.extract_text() or "" for p in doc.pages])

        (Path(output_folder) / f"{pdf_file.stem}.txt").write_text(text)
        n += 1
    return n