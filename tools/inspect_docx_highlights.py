"""List highlighted text in a Word document for report completion."""

import sys

from docx import Document


def main(path: str) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    document = Document(path)
    print(f"Paragraphs: {len(document.paragraphs)} Tables: {len(document.tables)}")
    for index, paragraph in enumerate(document.paragraphs):
        highlights = [run.text for run in paragraph.runs if run.font.highlight_color is not None]
        if highlights:
            print(f"P {index}: {paragraph.text!r}\n  HIGHLIGHT={highlights}")
    for table_index, table in enumerate(document.tables):
        for row_index, row in enumerate(table.rows):
            for cell_index, cell in enumerate(row.cells):
                highlights = [
                    run.text
                    for paragraph in cell.paragraphs
                    for run in paragraph.runs
                    if run.font.highlight_color is not None
                ]
                if highlights:
                    print(
                        f"T {table_index} R {row_index} C {cell_index}: {cell.text!r}\n"
                        f"  HIGHLIGHT={highlights}"
                    )


if __name__ == "__main__":
    main(sys.argv[1])
