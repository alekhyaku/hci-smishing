import csv
import os
import ast
import json
import pandas as pd
# Define the target quote
target_quote = "The USPS package has arrived at the warehouse and cannot be delivered due to incomplete address information."

# Initialize variables
matched_lines = []
quote_found = False  # Flag to indicate if the target quote has been found
# Define the path to your CSV file
folder_path = 'image_results'
csv_file_name = 'image_1_results.csv'  # Replace with your actual CSV file name
csv_file_path = os.path.join('..', folder_path, csv_file_name)

def load_csv(file_path):
    """
    Loads the CSV containing the word bounding boxes and text.
    Returns a list of dictionaries with word details (word text, bounding polygon).
    """
    word_data = []

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            word_text = row['Word Text']
            line_text = row['Line Text']
            word_bounding_polygon = ast.literal_eval(row['Word Bounding Polygon'])  # Safely parse the list of dictionaries
            confidence = float(row['Confidence'])
            
            word_data.append({
                'word': word_text,
                'line': line_text,
                'bounding_polygon': word_bounding_polygon,
                'confidence': confidence
            })
    
    return word_data

def find_bounding_boxes_for_string(input_string, word_data):
    """
    Matches the input string to the words in word_data sequentially,
    and returns the bounding boxes for each matched word.
    """
    input_words = input_string.split()  # Split input string into words
    print(f"Input words: {input_words}")
    bounding_boxes = []
    input_word_index = 0
    word = []
    line = []
    word_data = word_data[2:]

    for data in word_data:
        # Match words sequentially
        print("dataWord", data['word'].lower(), "inputWord", input_words[input_word_index].lower())
        print("input_word_index", input_word_index, "len(input_words)", len(input_words))
        print("input_words[input_word_index].lower().startswith(data['word'].lower())", input_words[input_word_index].lower().startswith(data['word'].lower()))
        if input_word_index < len(input_words) and data['word'].lower() == input_words[input_word_index].lower():
            word.append(data['word'])
            line.append(data['line'])
            bounding_boxes.append(data['bounding_polygon'])
            input_word_index += 1
        elif input_word_index < len(input_words) and input_words[input_word_index].lower().startswith(data['word'].lower()):
            # Handle split words
            remaining_word = input_words[input_word_index][len(data['word']):]
            remaining_word = input_words[input_word_index][len(data['word']):]
            if remaining_word and input_word_index + 1 < len(input_words) and remaining_word[0].lower() != input_words[input_word_index + 1][0].lower():
                print("remaining_word[0].lower()", remaining_word[0].lower(), "data['word'][0].lower()", data['word'][0].lower())
                continue  # Skip if the first character of the remaining word does not match the first character of the current word
            word.append(data['word'])
            line.append(data['line'])
            bounding_boxes.append(data['bounding_polygon'])
            input_words[input_word_index] = remaining_word
            print("remaining_word", remaining_word, "input_words", input_words[input_word_index])
            if not remaining_word:
                input_word_index += 1    
        if input_word_index == len(input_words):  # If all words have been matched
            break

    # Get the first and last word line data of the input string
    first_word_line = line[0]
    last_word_line = line[-1]
    first_word_lines = []
    middle_lines = []
    last_word_lines = []
    if first_word_line == last_word_line:
        for i in range(len(line)):
            first_word_lines.append(bounding_boxes[i])
    else:
        for i in range(len(line)):
            if line[i] == first_word_line:
                first_word_lines.append(bounding_boxes[i])
            elif line[i] == last_word_line:
                last_word_lines.append(bounding_boxes[i])
            else:
                middle_lines.append(bounding_boxes[i])

    return bounding_boxes, first_word_lines, middle_lines, last_word_lines

def adjust_bounding_box(bounding_boxes, first_word_lines, middle_lines, last_word_lines):
    """
    Adjust the bounding box based on the first and last matched word's coordinates.
    The left and right boundaries are defined by the first and last word's x-coordinates,
    and the top and bottom boundaries are the vertical limits of the matched words.
    """
    if not bounding_boxes:
        return None

    def get_extreme_coordinates(boxes):
        left_x = float('inf')
        right_x = -float('inf')
        top_y = float('inf')
        bottom_y = -float('inf')
        for bbox in boxes:
            for point in bbox:
                left_x = min(left_x, point['x'])
                right_x = max(right_x, point['x'])
                top_y = min(top_y, point['y'])
                bottom_y = max(bottom_y, point['y'])
        return left_x, right_x, top_y, bottom_y

    first_left_x, first_right_x, first_top_y, first_bottom_y = get_extreme_coordinates(first_word_lines)
    last_left_x, last_right_x, last_top_y, last_bottom_y = get_extreme_coordinates(last_word_lines)

    if not middle_lines and not last_word_lines:  # If there are no middle lines and no last word lines
        adjusted_bounding_box = [
            {'x': first_left_x, 'y': first_top_y},
            {'x': first_right_x, 'y': first_top_y},
            {'x': first_right_x, 'y': first_bottom_y},
            {'x': first_left_x, 'y': first_bottom_y}
        ]
    elif not middle_lines:  # If there are no middle lines
        adjusted_bounding_box = [
            {'x': first_left_x, 'y': first_top_y},
            {'x': first_right_x, 'y': first_top_y},
            {'x': first_right_x, 'y': first_bottom_y},
            {'x': last_right_x, 'y': last_top_y},
            {'x': last_right_x, 'y': last_bottom_y},
            {'x': last_left_x, 'y': last_bottom_y},
            {'x': last_left_x, 'y': last_top_y},
            {'x': first_left_x, 'y': first_bottom_y},
            {'x': first_left_x, 'y': first_top_y}
        ]
    else:  # Multiple lines case
        middle_left_x, middle_right_x, middle_top_y, middle_bottom_y = get_extreme_coordinates(middle_lines)
        adjusted_bounding_box = [
            {'x': first_left_x, 'y': first_top_y},
            {'x': first_right_x, 'y': first_top_y},
            {'x': first_right_x, 'y': first_bottom_y},
            {'x': middle_right_x, 'y': middle_top_y},
            {'x': middle_right_x, 'y': middle_bottom_y},
            {'x': last_right_x, 'y': last_top_y},
            {'x': last_right_x, 'y': last_bottom_y},
            {'x': last_left_x, 'y': last_bottom_y},
            {'x': last_left_x, 'y': last_top_y},
            {'x': middle_left_x, 'y': middle_bottom_y},
            {'x': middle_left_x, 'y': middle_top_y},
            {'x': first_left_x, 'y': first_bottom_y},
            {'x': first_left_x, 'y': first_top_y}
        ]

    return adjusted_bounding_box


def save_bounding_boxes_to_csv(bounding_boxes, output_csv_path, image_name, quote_label):
    """
    Saves the bounding boxes to a CSV file.
    """
    file_exists = os.path.isfile(output_csv_path)

    with open(output_csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['image_name', 'quote_label', 'bounding_box'])  # Write the header if the file does not exist
        for box in bounding_boxes:
            bounding_box_str = json.dumps(box)  # Convert the bounding box to a JSON string
            writer.writerow([image_name, quote_label, bounding_box_str])

def process_examples_csv(examples_csv_path, image_results_folder, output_folder):
    """
    Processes the examples.csv file and saves bounding boxes for quote_1 and quote_2 for each image.
    """
    examples_df = pd.read_csv(examples_csv_path)
    print("Examples CSV file read successfully")  # Debugging print statement
    print(examples_df.head())  # Print the first few rows of the DataFrame

    for index, row in examples_df.iterrows():
        image_name = row['Image_Name']
        # remove the suffix from the image name
        image_name = image_name.split('.')[0]
        quote_1 = row['Quote_1']
        quote_2 = row['Quote_2']
        sender = row["Sender"]
        is_sender = row["is_sender"]
        # Remove surrounding quotes from the strings
        quote_1 = quote_1.strip('\'"')
        quote_2 = quote_2.strip('\'"')
        # print(quote_1)
        # print("quote_2: ", quote_2)
        
        # Construct the path to the image results CSV file
        csv_file_name = f"{image_name}_results.csv"
        csv_file_path = os.path.join('..', image_results_folder, csv_file_name)
        
        if not os.path.exists(csv_file_path):
            print(f"CSV file for {image_name} does not exist. Skipping.")
            print(f"Expected path: {csv_file_path}")
            continue
        
        # Load the word data from the CSV file
        word_data = load_csv(csv_file_path)
        # print(word_data)
        
        # Find bounding boxes for quote_1 and quote_2
        bounding_boxes_1, first_word_lines_1, middle_lines_1, last_word_lines_1 = find_bounding_boxes_for_string(quote_1, word_data)
        bounding_boxes_2, first_word_lines_2, middle_lines_2, last_word_lines_2 = find_bounding_boxes_for_string(quote_2, word_data)
        print(f"Bounding boxes for quote_1: {bounding_boxes_1}")
        print(f"lines for quote_1: {first_word_lines_1}")
        print(f"lines for quote_1: {middle_lines_1}")
        print(f"lines for quote_1: {last_word_lines_1}")
        print(f"Bounding boxes for quote_2: {bounding_boxes_2}")
        print(f"lines for quote_2: {first_word_lines_2}")
        print(f"lines for quote_2: {middle_lines_2}")
        print(f"lines for quote_2: {last_word_lines_2}")
        # Adjust the bounding boxes
        adjusted_bounding_box_1 = adjust_bounding_box(bounding_boxes_1, first_word_lines_1, middle_lines_1, last_word_lines_1)
        adjusted_bounding_box_2 = adjust_bounding_box(bounding_boxes_2, first_word_lines_2, middle_lines_2, last_word_lines_2)
        # print(f"Adjusted bounding box for quote_1: {adjusted_bounding_box_1}")
        # print(f"Adjusted bounding box for quote_2: {adjusted_bounding_box_2}")
        # Save the bounding boxes to CSV files
        output_csv_path = os.path.join(output_folder, f"{image_name}_quote.csv")
        if adjusted_bounding_box_1:
            save_bounding_boxes_to_csv([adjusted_bounding_box_1], output_csv_path, image_name, 'quote_1')
            print(f"Bounding boxes for quote_1 saved to {output_csv_path}")
        
        if adjusted_bounding_box_2:
            save_bounding_boxes_to_csv([adjusted_bounding_box_2], output_csv_path, image_name, 'quote_2')
            print(f"Bounding boxes for quote_2 saved to {output_csv_path}")
        
        if is_sender == "Yes":
            sender_bounding_boxes, sender_first_line, middle_send_line, last_send_line = find_bounding_boxes_for_string(sender, word_data)
            print(f"Bounding boxes for sender: {sender_bounding_boxes}")
            print(f"Bounding boxes for sender: {sender_first_line}")
            print(f"Bounding boxes for sender: {middle_send_line}")
            print(f"Bounding boxes for sender: {last_send_line}")
            adjusted_sender_bounding_box = adjust_bounding_box(sender_bounding_boxes,sender_first_line, middle_send_line, last_send_line)
            save_bounding_boxes_to_csv([adjusted_sender_bounding_box], output_csv_path, image_name, 'sender')

def main():
    # Define the paths
    examples_csv_path = os.path.join('gallery_explan', 'examples.csv')
    image_results_folder = 'image_results'
    output_folder = 'quote_results'
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Process the examples.csv file
    process_examples_csv(examples_csv_path, image_results_folder, output_folder)

if __name__ == "__main__":
    main()
