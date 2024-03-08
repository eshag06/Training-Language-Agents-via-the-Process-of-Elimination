import re
import numpy as np
import os


def split_text(text):
    # Define regex patterns for each section
    query_pattern = r"Query:\s*(.*?)\s*Generation:"
    generation_pattern = r"Generation:\s*(.*?)\s*Ground Truth Answer:"
    ground_truth_pattern = r"Ground Truth Answer:\s*(.*?)\s*(Logits:|$)"
    logits_pattern = r"Logits:\s*(.*)"

    # Extract each part
    query_match = re.search(query_pattern, text, re.DOTALL)
    generation_match = re.search(generation_pattern, text, re.DOTALL)
    ground_truth_match = re.search(ground_truth_pattern, text, re.DOTALL)
    logits_match = re.search(logits_pattern, text, re.DOTALL)

    query = query_match.group(1) if query_pattern else ""
    generation = generation_match.group(1) if generation_match else ""
    ground_truth = ground_truth_match.group(1) if ground_truth_match else ""
    logits = logits_match.group(1) if logits_match else ""

    return query, generation, ground_truth, logits


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
    model_name, num_shots, dataset, decode_method, num_iter = parameters[1], parameters[2], parameters[3], parameters[
        4], parameters[5]

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

model_name = []
num_shots = []
decode_method = []
acc = []
datasets_name = []

# List all files in the 'results' directory
files = os.listdir(results_dir)
for file in files:
    # Construct the full file path
    file_path = os.path.join(results_dir, file)
    parameters = decompress_file_name(file_path)

    if len(parameters) == 6 and (parameters[3] == 'beam' or parameters[3] == 'nucleus'):
        decode_method.append(parameters[3] + ' ' + parameters[5])
    else:
        decode_method.append(parameters[3])
    model_name.append(parameters[0])
    num_shots.append(parameters[1])
    datasets_name.append(parameters[2])

    # Check if it's a file and not a directory
    if os.path.isfile(file_path):
        # Open and read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

    time, content = remove_last_line(content)
    generations = content.split('Next:')[1:]

    correct = 0
    for generation in generations:
        query, gen, ground_truth, logits = split_text(generation)
        if ground_truth == gen[0]:
            correct += 1
    if len(generations) == 0:
        acc.append(0)
    else:
        acc.append(correct / len(generations))

for i in range(len(model_name)):
    print(f'{datasets_name[i]:<20} | {decode_method[i]:<20} | {num_shots[i]:<20} | {acc[i]:<20}')
