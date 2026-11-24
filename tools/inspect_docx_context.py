"""Print report paragraphs and tables with indexes for structured DOCX edits."""

import sys

from docx import Document


def main(path: str) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    document = Document(path)
    for index, paragraph in enumerate(document.paragraphs):
        if paragraph.text.strip():
            print(f"P{index}: {paragraph.text}")
    for table_index, table in enumerate(document.tables):
        print(f"\nTABLE {table_index}")
        for row_index, row in enumerate(table.rows):
            print(f"R{row_index}: " + " | ".join(cell.text.replace("\n", " / ") for cell in row.cells))


if __name__ == "__main__":
    main(sys.argv[1])
