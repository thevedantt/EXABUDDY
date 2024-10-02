import os
from flask import Flask, request, render_template_string
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import re

# Set the backend (uncomment one based on your installation)
# os.environ["USE_TF"] = "1"  # If using TensorFlow
os.environ["USE_TORCH"] = "1"  # If using PyTorch

app = Flask(__name__)

# Load the OCR model
ocr_model = ocr_predictor(pretrained=True)

# Allowed extensions for file upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

# Function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Simple function for basic text post-processing (could be expanded with NLP or AI)
def clean_and_format_text(extracted_text):
    # Remove extra spaces and line breaks intelligently
    extracted_text = re.sub(r'\n+', '\n', extracted_text)  # Remove multiple newlines
    extracted_text = re.sub(r'\s+', ' ', extracted_text)  # Replace multiple spaces with a single space

    # Here we could add NLP-based post-processing like spell checking, grammar correction, etc.
    # For example, if integrating a library like `language_tool_python` for text correction:
    # tool = language_tool_python.LanguageTool('en-US')
    # extracted_text = tool.correct(extracted_text)

    return extracted_text

# Route to upload an image or PDF and process OCR, display result as formatted paragraphs
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']

    if file.filename == '':
        return "No selected file", 400

    if file and allowed_file(file.filename):
        try:
            # Check if the file is an image or PDF
            file_ext = file.filename.rsplit('.', 1)[1].lower()

            if file_ext in ['png', 'jpg', 'jpeg']:
                # Process as image
                doc = DocumentFile.from_images(file)
            elif file_ext == 'pdf':
                # Process as PDF
                doc = DocumentFile.from_pdf(file)

            # Perform OCR on the document
            result = ocr_model(doc)

            # Extract plain text from the prediction result
            raw_text = "\n".join(
                [word.value for page in result.pages for block in page.blocks for line in block.lines for word in line.words]
            )

            # Clean and format the extracted text
            formatted_text = clean_and_format_text(raw_text)

            # Display the extracted text as paragraphs on the webpage
            return render_template_string('''
                <h1>OCR Result</h1>
                <p>{{ formatted_text }}</p>
                <br>
                <a href="/">Go back</a>
                ''', formatted_text=formatted_text)

        except Exception as e:
            return f"Error: {str(e)}", 500
    else:
        return "File type not allowed", 400

# Home route with upload form
@app.route('/')
def index():
    return '''
    <h1>OCR with DocTR (Image and PDF Support)</h1>
    <p>Upload an image or PDF containing handwritten or printed text to extract the text using OCR.</p>
    <form method="post" enctype="multipart/form-data" action="/upload">
        <input type="file" name="file">
        <input type="submit" value="Upload">
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)
