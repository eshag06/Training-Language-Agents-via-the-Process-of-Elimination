import re
import numpy as np
import os


def extract_lines_with_re(text):
    # Find all lines containing the pattern
    pattern = "The correct answer is"
    matching_lines = re.findall(r'^.*' + re.escape(pattern) + r'.*$', text, re.MULTILINE)

    return matching_lines


def remove_last_line(text):
    # Split the text into lines
    lines = text.split('\n')

    # Extract the last line
    last_line = lines[-1]

    # Rejoin the remaining lines
    new_text = '\n'.join(lines[:-1])

    return last_line, new_text


def decompress_file_name(file_name):
    parameters = file_name[:-4].split('_')
    model_name, num_shots, dataset, decode_method, num_iter = parameters[1], parameters[2], parameters[3], parameters[4], parameters[5]

    if decode_method == 'beam':
        num_beams = parameters[6]
        print(f"""
Parameter setting:

Model: {model_name}
Number of shots: {num_shots}
Dataset: {dataset}
Decode method: {decode_method}
Beam width: {num_beams}
Number of iterations (queries): {num_iter}
        """)
        return model_name, num_shots, dataset, decode_method, num_iter, num_beams
    elif decode_method == 'nucleus':
        top_p = parameters[6]
        print(f"""
Parameter setting:

Model: {model_name}
Number of shots: {num_shots}
Dataset: {dataset}
Decode method: {decode_method}
top_p: {top_p}
Number of iterations (queries): {num_iter}
        """)
        return model_name, num_shots, dataset, decode_method, num_iter, top_p

    print(f"""
Parameter setting:

Model: {model_name}
Number of shots: {num_shots}
Dataset: {dataset}
Decode method: {decode_method}
Number of iterations (queries): {num_iter}
    """)
    return model_name, num_shots, dataset, decode_method, num_iter



# Path to the 'results' directory
results_dir = './results'

# List all files in the 'results' directory
files = os.listdir(results_dir)
for file in files[:-1]:
    # Construct the full file path
    file_path = os.path.join(results_dir, file)
    parameters = decompress_file_name(file_path)

    # Check if it's a file and not a directory
    if os.path.isfile(file_path):
        # Open and read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

    time, content = remove_last_line(content)
    generations = content.split('Next:')[1:]

    correct_answers = [generation[-2] for generation in generations]
    print(time)

    correct = 0
    for i, generation in enumerate(generations):
        resulting_lines = extract_lines_with_re(generation)
        if len(resulting_lines) == 0:
            continue
        else:
            if resulting_lines[0][-1] == correct_answers[i]:
                correct += 1

    print(f'The accuracy is: {correct / len(correct_answers)}')
    print('-' * 100)

