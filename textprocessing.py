import pandas as pd
import ast
import sys
import sqlite3
import psycopg2
import os

# Database connection details
DB_HOST = os.environ.get('DB_HOST')
DB_NAME = os.environ.get('DB_NAME')
DB_USER = os.environ.get('DB_USER')
DB_PASSWORD = os.environ.get('DB_PASSWORD')

def load_csv(image_name):
    data = pd.read_csv(image_name, header='infer') 
    #display(data)
    return data

# Now we want to iterate over the data and add each x and y value array to the new columns
def create_xycol(data):
    xy_col = pd.DataFrame(columns = ['x_arr', 'y_arr'])
    for index, row in data.iterrows():
        bounding_string = row["Bounding Box"]

        # Convert the string to a Python object (list of dictionaries)
        bounding_obj = ast.literal_eval(bounding_string)

        # Extract x and y values into separate lists
        x_values = [point['x'] for point in bounding_obj]
        y_values = [point['y'] for point in bounding_obj]

        df = pd.DataFrame({'x_arr': [x_values], 'y_arr': [y_values]})

        # Add x and y to the new columns
        xy_col = pd.concat([xy_col, df], ignore_index = True)
    return xy_col

def merge_data_xy(data, xy_col):

    # Now using the bounding Boxes we want to join xy_col with the data dataframe
    data = pd.concat([data, xy_col], axis=1)

    # Then we can drop the Bounding box column
    data.drop('Bounding Box', axis=1, inplace=True)
    #display(data)
    return data

# Next we want to remove all the lines that have y_arr less than 300
# Function to check if all values in y_values are less than 300
def check_y_values(row):
    return all(y < 300 for y in row)

def filter_30(data):
    # Apply the function to the 'y_values' column and filter rows
    df_filtered = data[~data['y_arr'].apply(check_y_values)]

    # Display the filtered DataFrame
    #display(df_filtered)
    return df_filtered

def remove_bottom(df_filtered):
    # Next we want to get rid of the junk on the bottom
    # So we check first if there is a "The sender is not in your contact list", and if that line exists remove the indexes from that line onwards
    # Find the index where 'line_text' contains the message
    message = "The sender is not in your contact list"
    index_to_remove_from = df_filtered[df_filtered['Line Text'].str.contains(message, case=False)].index

    # Drop the rows from that index onwards
    if not index_to_remove_from.empty:
        index = index_to_remove_from[0]
        # Drop all rows from this index onwards
        df_filtered = df_filtered.loc[:index-1] 

    # Display the updated DataFrame
    #display(df_filtered)
    return df_filtered

# if that line does not exist we want to remove all lines with y values  > 2300
# Function to check if all values in y_values are greater than 2300
def check_y_values_greater(row):
    return all(y > 2300 for y in row)

def remove_extra(df_filtered):
    # Apply the function to the 'y_values' column and filter rows
    df_filtered = df_filtered[~df_filtered['y_arr'].apply(check_y_values)]


    # Display the filtered DataFrame
    df_filtered = df_filtered.reset_index(drop=True)
    #display(df_filtered)
    return df_filtered

# Next we want to remove the inbetween junk from the sender and the actual message
# All of the actual message will have an initial x_arr value < 100. So everything between the first row and the first row where the first value of the x_arr can be removed
# Rarely, there will a random line in between the number and the actual message that is ".", that is <100 so check to ignore it and remove that line

def remove_date(df_filtered):
    # Initialize list to track rows we want to keep
    rows_to_keep = [0]  # Start by keeping the first row (index 0)

    # Iterate through the DataFrame starting from the second row (index 1)
    for idx, row in df_filtered.iloc[1:].iterrows():
        # If the first value in 'x_values' is less than 100 and it's not a "."
        if row['x_arr'][0] < 100 and row['Line Text'] != ".":
            rows_to_keep.append(idx)

    # Filter the DataFrame to keep only the selected rows
    df_filtered = df_filtered.loc[rows_to_keep]

    # sometimes the sender will have " >" attatched to the end of the string
    
    # Remove the substring " >" from the 'column_name' column
    df_filtered['Line Text'] = df_filtered['Line Text'].str.replace(' >', '', regex=False)

    # Display the filtered DataFrame
    #display(df_filtered)
    
    return df_filtered

def filter_single(image_name):
    data = load_csv(image_name)
    xy_col = create_xycol(data)
    #display(xy_col)
    data = merge_data_xy(data, xy_col)
    df_filtered = filter_30(data)
    df_filtered = remove_bottom(df_filtered)
    df_filtered = remove_extra(df_filtered)
    df_filtered = remove_date(df_filtered)
    return df_filtered

def text_process(image_name):
    data = filter_single(image_name)
    sender = data.iloc[0]['Line Text']

    # Step 2: Concatenate the remaining rows' line_text into the message (excluding the first row)
    message = " ".join(data.iloc[1:]['Line Text'])

    # Step 3: Create the new DataFrame
    df_message = pd.DataFrame({
        'Sender': [sender],
        'Text_Message': [message]
    })

    return df_message

def filter_single_db(image_name):
    # instead of loading from a csv we will load from a database
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    data = pd.read_sql_query("SELECT * FROM lineresults WHERE filename = ?", conn, params=(image_name,))
    all_data = pd.read_sql_query("SELECT * FROM lineresults", conn)
    all_data_result = pd.read_sql_query("SELECT * FROM results", conn)
    # That should return all the rows with the filename
    conn.close()
    # Rename columns bounding_box to Bounding Box and line_text to Line Text
    data = data.rename(columns={"bounding_box": "Bounding Box", "line_text": "Line Text"})
    print("DATA: ",data)
    print("ALL lineresults DATA: ", all_data)
    print("All results data: ",all_data_result)
    xy_col = create_xycol(data)
    #display(xy_col)
    data = merge_data_xy(data, xy_col)
    df_filtered = filter_30(data)
    df_filtered = remove_bottom(df_filtered)
    df_filtered = remove_extra(df_filtered)
    df_filtered = remove_date(df_filtered)
    return df_filtered

def text_process_db(image_name):
    data = filter_single_db(image_name)
    sender = data.iloc[0]['Line Text']

    # Step 2: Concatenate the remaining rows' line_text into the message (excluding the first row)
    message = " ".join(data.iloc[1:]['Line Text'])

    # Step 3: Create the new DataFrame
    df_message = pd.DataFrame({
        'Sender': [sender],
        'Text_Message': [message]
    })
    print("DF MESSAGE: ",df_message)
    return df_message

if __name__ == "__main__":
    text_process(sys.argv[1])
