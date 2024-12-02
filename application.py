from flask import Flask, render_template, request, send_from_directory, session, send_file
from werkzeug.utils import secure_filename
import random
import os
import io
import psycopg2
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from textprocessing import text_process, text_process_db  # Import the function from the other script
import pandas as pd
import ast
import sqlite3
import json
import numpy as np
# from genexpla import generate_explanation

app = Flask(__name__)
app.secret_key = 'secret'  # Set a secret key for session management
status_message = ""
IMAGES_FOLDER = 'images'
app.config['IMAGES_FOLDER'] = IMAGES_FOLDER
app.config['GALLERY_FOLDER'] = 'gallery_img'

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Database connection details
DB_HOST = os.environ.get('DB_HOST')
DB_NAME = os.environ.get('DB_NAME')
DB_USER = os.environ.get('DB_USER')
DB_PASSWORD = os.environ.get('DB_PASSWORD')

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_db():
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            filename TEXT NOT NULL PRIMARY KEY,
            data BYTEA NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id SERIAL PRIMARY KEY,
            filename TEXT NOT NULL,
            line_text TEXT,
            bounding_box JSON,
            word_text TEXT,
            word_bounding_polygon JSON,
            confidence REAL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS lineresults (
            id SERIAL PRIMARY KEY,
            filename TEXT NOT NULL,
            line_text TEXT,
            bounding_box JSON
        )
    ''')
    conn.commit()
    conn.close()


def store_image_in_db(filename, data):
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    cursor = conn.cursor()
    cursor.execute('INSERT INTO images (filename, data) VALUES (%s, %s)', (filename, data))
    conn.commit()
    conn.close()

# Retrieve image from the database
def get_image_from_db(filename):
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    cursor = conn.cursor()
    cursor.execute('SELECT data FROM images WHERE filename = %s', (filename,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return row[0]
    return None

def clear_db():
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    cursor = conn.cursor()
    cursor.execute('DELETE FROM images')
    cursor.execute('DELETE FROM results')
    cursor.execute('DELETE FROM lineresults')
    conn.commit()
    conn.close()

def image_to_text(filename):
     # Retrieve the image data from the database
    image_data = get_image_from_db(filename)
    if not image_data:
        print("Image not found in the database")
        return

    print("Image retrieved successfully from the database!")

    # Check if the image name already has a result in the results database
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM results WHERE filename = %s', (filename,))
    if cursor.fetchone():
        print("Results already exist for this image")
        return

    # Set the values of your computer vision endpoint and computer vision key
    # as environment variables:
    try:
        endpoint = os.environ["VISION_ENDPOINT"]
        key = os.environ["VISION_KEY"]
    except KeyError:
        print("Missing environment variable 'VISION_ENDPOINT' or 'VISION_KEY'")
        print("Set them before running this sample.")
        exit()

    # Create an Image Analysis client
    client = ImageAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
    )

    # Analyze the image data
    result = client.analyze(io.BytesIO(image_data), visual_features=[VisualFeatures.READ])

    # Process the result
    process_result(result, filename)

def process_result(result, filename):
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    cursor = conn.cursor()
    # Example processing logic
    if result.read is not None:
        for line in result.read.blocks[0].lines:
            print(f"   Line: '{line.text}', Bounding box {line.bounding_polygon}")
            for word in line.words:
                cursor.execute('''
                    INSERT INTO results (filename, line_text, bounding_box, word_text, word_bounding_polygon, confidence)
                    VALUES (%s, %s, %s, %s, %s, %s)
                ''', (
                    filename,
                    line.text,
                    str(line.bounding_polygon),
                    word.text,
                    str(word.bounding_polygon),
                    word.confidence
                ))
                print(f"     Word: '{word.text}', Bounding polygon {word.bounding_polygon}, Confidence {word.confidence:.4f}")
    conn.commit()

    if result.read is not None:
        for line in result.read.blocks[0].lines:
            cursor.execute('''
                INSERT INTO lineresults (filename, line_text, bounding_box)
                VALUES (%s, %s, %s)
            ''', (
                filename,
                line.text,
                str(line.bounding_polygon)
            ))
    conn.commit()
    conn.close()

def get_processed_quotes(image_name):
    # Main program
    quotes_csv = os.path.join('quote_results', f"{image_name}_quote.csv")
    # Read the bounding boxes for each quote

    print(f"Bounding boxes calculated and saved to {output_csv}.")

@app.route("/", methods=["GET", "POST"])
def home():
    global status_message
    status_message = ""
    message = {}  # Initialize message as an empty dictionary
    filename = ""
    session.pop('message', None)
    if request.method == "POST":
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # file_path = os.path.join(app.config['IMAGES_FOLDER'], filename)
            file_data = file.read()
            if get_image_from_db(filename):
                status_message = "File with the same name already exists. Please rename your file and try again."
            else:
                store_image_in_db(filename, file_data)
                image_to_text(filename)
                # line_csv_filename = os.path.join(app.config['RESULTS_FOLDER'], f"{os.path.splitext(filename)[0]}_line_results.csv")
                df = text_process_db(filename)  # Call the function and get the DataFrame
                message = df.iloc[0].to_dict()  # Convert the single row DataFrame to a dictionary
                session['message'] = message  # Store the message in the session
                status_message = "File uploaded successfully!"
        else:
            status_message = "Invalid file type or no file uploaded"
    return render_template("index.html", status_message=status_message, message=message, filename=filename)

# Get the quotes to highlight and their bounding boxes
# get_processed_quotes()

@app.route("/gallery")
def gallery():
    print("Gallery route accessed")  # Debugging print statement
    gallery_imgs = os.listdir(app.config['GALLERY_FOLDER'])[:20]
    # print(f"Gallery images: {gallery_imgs}")  # Debugging print statement
    # Randomize the order of the images
    random.shuffle(gallery_imgs)
    # print(f"Gallery images after shuffling: {gallery_imgs}")  # Debugging print statement
    # Read the csv file to get the image bounding boxes
    quote_path = os.path.join('quote_results', 'image_1_quote.csv')

    # Read the examples.csv file
    examples_csv_path = os.path.join('gallery_explan', 'examples.csv')
    df = pd.read_csv(examples_csv_path)
    print("CSV file read successfully")  # Debugging print statement
    # print(df.head())  # Print the first few rows of the DataFrame

    # Create a dictionary to store classification and explanation for each image
    image_info = {}
    for image_file in gallery_imgs:
        # print(f"Processing image file: {image_file}")  # Debugging print statement
        # Convert both to lowercase and strip any leading/trailing whitespace
        image_file_clean = image_file.strip().lower()
        df["Image_Name"] = df["Image_Name"].str.strip().str.lower()
        # Search for the image name in the "Image_Name" column
        row = df[df["Image_Name"] == image_file_clean]
        # print(f"Matching row for {image_file}: {row}")  # Debugging print statement
        if not row.empty:
            classification = row["Classification"].values[0]
            explanation = row["Explanation"].values[0]
            quote_1_ex = row["Quote_1_EX"].values[0]
            quote_2_ex = row["Quote_2_EX"].values[0]
            is_sender = row["is_sender"].values[0]
            sender_ex = row["is_sender_EX"].values[0]
            quote_1 = row["Quote_1"].values[0]
            quote_2 = row["Quote_2"].values[0]
            sender = row["Sender"].values[0]
            # get the bounding boxes for quote_1, quote_2, and sender
            # read from csv image_i_quote.csv
            # remove the suffix from the image name
            image_name = os.path.splitext(image_file_clean)[0]
            quote_path = os.path.join('quote_results', f"{image_name}_quote.csv")
            # print(f"Reading bounding boxes fro m {quote_path}")  # Debugging print statement
            quote_df = pd.read_csv(quote_path)
            # print(quote_df)
            # Get the bounding boxes for quote_1, quote_2, and sender
            quote_1_bound = np.array(ast.literal_eval(quote_df[quote_df["quote_label"] == "quote_1"]["bounding_box"].values[0]))
            quote_2_bound = np.array(ast.literal_eval(quote_df[quote_df["quote_label"] == "quote_2"]["bounding_box"].values[0]))
            if is_sender == "Yes":
                sender_bound = np.array(ast.literal_eval(quote_df[quote_df["quote_label"] == "sender"]["bounding_box"].values[0]))
            else:
                sender_bound = None
            # Convert bounding boxes to lists to ensure they are JSON serializable
            if quote_1_bound is not None:
                quote_1_bound = quote_1_bound.tolist()
                # quote_1_bound = [list(point.values()) for point in quote_1_bound]
            if quote_2_bound is not None:
                # quote_2_bound = [list(point.values()) for point in quote_2_bound]
                quote_2_bound = quote_2_bound.tolist()
            if sender_bound is not None:
                # sender_bound = [list(point.values()) for point in sender_bound]
                sender_bound = sender_bound.tolist()

            # print(f"Bounding boxes for {image_file}: quote_1: {quote_1_bound}, quote_2: {quote_2_bound}, sender: {sender_bound}")  # Debugging print statement
            image_info[image_file] = {
                "classification": classification,
                "explanation": explanation,
                "quote_1_bound": quote_1_bound,
                "quote_2_bound": quote_2_bound,
                "sender_bound": sender_bound,
                "is_sender": is_sender,
                "quote_1_ex": quote_1_ex,
                "quote_2_ex": quote_2_ex,
                "sender_ex": sender_ex,
                "quote_1": quote_1,
                "quote_2": quote_2,
                "sender": sender
            }

    # print(image_info)
    return render_template("gallery.html", gallery_imgs=gallery_imgs, image_info=image_info)

@app.route("/explain/<filename>")
def explain(filename):
    message = session.get('message', {})  # Retrieve the message from the session
    classification = "Example Classification"
    explanation = message
    # explanation = generate_explanation(message)  # Generate explanation using the message dictionary
    return render_template("explain.html", filename=filename, classification=classification, explanation=explanation)

@app.route('/images/<filename>')
def serve_image(filename):
    file_data = get_image_from_db(filename)
    if file_data:
        return send_file(io.BytesIO(file_data), mimetype='image/jpeg', as_attachment=True, attachment_filename=filename)
    return "Image not found", 404

@app.route('/gallery_img/<filename>')
def serve_gallery_image(filename):
    return send_from_directory(app.config['GALLERY_FOLDER'], filename)

# Initialize the db
init_db()

if __name__ == "__main__":
    app.run(debug=True)
