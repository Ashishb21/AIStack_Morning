import PyPDF2
import os
import pandas as pd

file_path = "/Users/ashishbansal/Documents/Training/AiStack/Aistack_Course/Project_1/2.RAG/1.Process_Data/datasets/pdf_files/2023_Jan_7_Feature_Engineering_Techniques.pdf"

with open(file_path, "rb") as file:
    reader = PyPDF2.PdfReader(file)

    # Initialize an empty string to store the extracted text
    list_of_pages = []
    page_counter = 1

    for page in reader.pages:
        page_dict = {
            "file_name": reader.metadata.get("/Title"),
            "producer": reader.metadata.get("/Producer"),
            "page_number": page_counter,
            "text": page.extract_text(),
            "images": page.images,
        }

        list_of_pages.append(page_dict)

        page_counter += 1

# Convert the list of pages to a pandas DataFrame
pages_df = pd.DataFrame(list_of_pages)
print(pages_df)

pages_df.to_csv("parsed_pdf.csv")