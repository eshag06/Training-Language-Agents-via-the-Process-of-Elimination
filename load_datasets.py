import json
import random


def load_commonsenseQA(modify_num_answer=''):
    file_path = "datasets/train_rand_split.jsonl"

    data = []  # List to store the JSON objects
    # Open the file and read line by line
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Parse the JSON object in each line
            json_object = json.loads(line)

            # Do something with the JSON object (here we're just appending it to a list)
            data.append(json_object)

    print(len(data))
    if modify_num_answer == 'reduce':
        return reduce_randomly_and_relabel_choices(data, 2)
    return data


def reduce_randomly_and_relabel_choices(data, n):
    for item in data:
        # Find the correct choice
        correct_choice = next(choice for choice in item['question']['choices'] if choice['label'] == item['answerKey'])

        # Get all incorrect choices
        incorrect_choices = [choice for choice in item['question']['choices'] if choice['label'] != item['answerKey']]

        # Shuffle the incorrect choices
        random.shuffle(incorrect_choices)

        # Select the first two incorrect choices
        selected_incorrect_choices = incorrect_choices[:n]

        # Combine correct choice with selected incorrect choices
        combined_choices = [correct_choice] + selected_incorrect_choices

        # Sort the choices by their original text
        combined_choices.sort(key=lambda x: x['text'])

        # Reassign labels as 'A', 'B', 'C', and update the answerKey if needed
        for i, choice in enumerate(combined_choices):
            new_label = chr(65 + i)  # 65 is ASCII for 'A'
            if choice['text'] == correct_choice['text']:
                item['answerKey'] = new_label
            choice['label'] = new_label

        # Update the choices in the item
        item['question']['choices'] = combined_choices

    return data
