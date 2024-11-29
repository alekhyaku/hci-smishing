from flask import Flask, render_template, request, send_from_directory, session, jsonify
from werkzeug.utils import secure_filename
import requests
import shutil
import random
import os
import csv
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from textprocessing import text_process  # Import the function from the other script
import pandas as pd
import ast
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

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_text(filename):
    # Ensure the images directory exists
    if not os.path.exists(app.config['IMAGES_FOLDER']):
        os.makedirs(app.config['IMAGES_FOLDER'])

    # Copy the image from the uploads directory to the images directory
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_path = os.path.join(app.config['IMAGES_FOLDER'], filename)
    shutil.copy(upload_path, image_path)

    print("Image copied successfully!")
    # print()

    print("Image downloaded successfully!")

    # check if the image name already has a result in the results folder
    if os.path.exists(os.path.join(app.config['RESULTS_FOLDER'], f"{os.path.splitext(filename)[0]}_results.csv")):
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

    # Load image to analyze into a 'bytes' object
    with open(image_path, "rb") as f:
        image_data = f.read()

    # Check if the image data is valid
    if not image_data:
        print("Invalid image data")
        return

    # Extract text (OCR) from an image stream. This will be a synchronously (blocking) call.
    result = client.analyze(
        image_data=image_data,
        visual_features=[VisualFeatures.READ]
    )

    # Print text (OCR) analysis results to the console
    print("Image analysis results:")
    print(" Read:")

     # Write OCR results to a CSV file
    csv_filename = os.path.join(app.config['RESULTS_FOLDER'], f"{os.path.splitext(filename)[0]}_results.csv")
    with open(csv_filename, "a", newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write header only if the file is empty
        if csvfile.tell() == 0:
            csvwriter.writerow(["Line Text", "Bounding Box", "Word Text", "Word Bounding Polygon", "Confidence"])
        if result.read is not None:
            for line in result.read.blocks[0].lines:
                print(f"   Line: '{line.text}', Bounding box {line.bounding_polygon}")
                for word in line.words:
                    csvwriter.writerow([
                        line.text,
                        line.bounding_polygon,
                        word.text,
                        word.bounding_polygon,
                        f"{word.confidence:.4f}"
                    ])
                    print(f"``     Word: '{word.text}', Bounding polygon {word.bounding_polygon}, Confidence {word.confidence:.4f}")

    # Write line results to a separate CSV file
    line_csv_filename = os.path.join(app.config['RESULTS_FOLDER'], f"{os.path.splitext(filename)[0]}_line_results.csv")
    with open(line_csv_filename, "a", newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write header only if the file is empty
        if csvfile.tell() == 0:
            csvwriter.writerow(["Line Text", "Bounding Box"])
        if result.read is not None:
            for line in result.read.blocks[0].lines:
                csvwriter.writerow([
                    line.text,
                    line.bounding_polygon
                ])


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
            file_path = os.path.join(app.config['IMAGES_FOLDER'], filename)
            if os.path.exists(file_path):
                status_message = "File with the same name already exists. Please rename your file and try again."
            else:
                file.save(f'{app.config["UPLOAD_FOLDER"]}/{filename}')
                image_to_text(filename)
                line_csv_filename = os.path.join(app.config['RESULTS_FOLDER'], f"{os.path.splitext(filename)[0]}_line_results.csv")
                df = text_process(line_csv_filename)  # Call the function and get the DataFrame
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
    return send_from_directory(app.config['IMAGES_FOLDER'], filename)

@app.route('/gallery_img/<filename>')
def serve_gallery_image(filename):
    return send_from_directory(app.config['GALLERY_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
