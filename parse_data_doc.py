import os
from docx import Document

file_path = "/Users/ashishbansal/Documents/Training/AiStack/Aistack_Course/Project_1/2.RAG/1.Process_Data/datasets/word_files/2023_Jan_7_Feature_Engineering_Techniques.docx"

doc = Document(file_path)

text = []
for paragraph in doc.paragraphs:
    text.append(paragraph.text)

full_text = "\n".join(text)
print(full_text)