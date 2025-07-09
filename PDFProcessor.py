import PyPDF2
import os
import csv
import re
import tiktoken 

class PDFProcessor:
    def __init__(self,folder_path, model='cl100k_base', token_limit= 5000):
        self.folder_path = folder_path
        self.model = model
        self.token_limit = token_limit
        self.tokenizer = tiktoken.get_encoding(self.model)

    def read_pdf(self, file_path):
        """
        Read and get text from a pdf file
        """

        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
        return text
    
    def clean_text(self, text):
        """
        clean the extracted text
        """

        text = text.lower()
        #Remove special characters and digits
        text = re.sub(r'[^a-z\s]', '', text)
        #REMOVE extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    def chunk_by_token_limit(self, text):
        tokens = self.tokenizer.encode(text)
        chunks = []
        current_chunk = []
        current_token_count = 0
        for token in tokens:
            current_token_count += 1
            current_chunk.append(token)
            if current_token_count >= self.token_limit:
                chunks.append(current_chunk)
                current_chunk = []
                current_token_count = 0
        if current_chunk:
            chunks.append(current_chunk)

        text_chunks = [self.tokenizer.decode(chunk) for chunk in chunks]
        return text_chunks
    
    def semantic_chunking(self, text):
        """
        Chunk thetextin into paragraphs or semantic sections.
        """

        #split the text by double newline
        chunks = text.split('\n\n')

        #further clean and ensure meaningful chunks

        cleaned_chunks = []
        for chunk in chunks:
            cleaned_chunk = chunk.strip()
            if len(cleaned_chunk) > 50:
                cleaned_chunks.append(cleaned_chunk)

        return cleaned_chunks
    
    def process_folder (self):
        data = []
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.pdf'):
                file_path = os.path.join(self.folder_path,filename)
                text =self.read_pdf(file_path=file_path)
                cleaned_text = self.clean_text(text=text)
                chunks = self.chunk_by_token_limit(cleaned_text)
                for i, chunk in enumerate(chunks):
                    data.append({'filename': filename, 'chunk_id': i, 'chunk' : chunk})
        return data
    
    def save_to_csv(self,data, output_csv):
        with open(output_csv, mode='w', newline = '', encoding ='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=['fieldname', 'chunk_id', 'chunk'])
            writer.writeheader()
            for row in data:
                writer.writerow(data)
    
    def process_and_save(self,output_csv):
        """
        Process PDF files and save the results to a csv file
        """

        data = self.process_folder()
        self.save_to_csv(data=data,output_csv=output_csv)

if __name__ == "__main__":
    folder_path = ''
    output_csv = ''
    pdf_processor = PDFProcessor(folder_path=folder_path)
    pdf_processor.process_and_save(output_csv=output_csv)
    
