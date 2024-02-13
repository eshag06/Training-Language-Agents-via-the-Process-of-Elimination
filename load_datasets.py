import json


def load_commonsenseQA():
    file_path = "datasets/train_rand_split.jsonl"

    data = []  # List to store the JSON objects
    # Open the file and read line by line
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Parse the JSON object in each line
            json_object = json.loads(line)

            # Do something with the JSON object (here we're just appending it to a list)
            data.append(json_object)

    return data

