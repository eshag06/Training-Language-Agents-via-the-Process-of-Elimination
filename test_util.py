import copy
import time
import numpy as np
import torch

from util import DatasetLoader
from util import Query
from util import Model
from util import ExperimentLogger

model = Model(model_name='meta-llama/Llama-2-7b-hf')


def modify_query_info_based_on_tokens(query_info, sorted_token_probs, k):
    probs = [prob for _, prob in sorted_token_probs if prob >= 0]
    average_prob = np.mean(probs)
    std_dev = np.std(probs)
    choices = query_info['final_question']["question"]["choices"]
    answer = query_info['final_question']['answerKey']
    correct_choice_text = next(choice["text"] for choice in choices if choice["label"] == answer)

    threshold = 1 * average_prob + k / 100 * std_dev
    print(sorted_token_probs)
    print(threshold)
    removed_choices = [choice.replace("▁", "") for choice, prob in sorted_token_probs if prob > threshold]
    if len(removed_choices) != 0:
        removed_choices = removed_choices[0]
    else:
        return query_info

    new_choices = [choice for choice in query_info['final_question']['question']['choices'] if
                   choice['label'] not in removed_choices]
    dict = {}
    answer_removed = True
    for i, choice in enumerate(new_choices):
        dict[choice['label']] = chr(65 + i)
        choice['label'] = chr(65 + i)
        if choice["text"] == correct_choice_text:
            updated_answer = choice["label"]
            answer_removed = False
    if answer_removed:
        updated_answer = 'Z'

    query_info['final_question']['question']['choices'] = new_choices
    query_info['final_question']['answerKey'] = updated_answer

    return query_info


def experiment(model, num_shots, dataset, decode_method, num_iter, num_beams=0, top_p=0.0):
    dataset_loader = DatasetLoader()
    experiment_logger = ExperimentLogger()

    data = dataset_loader.load_dataset(dataset)
    query_maker = Query(data, dataset)

    output_file_name = experiment_logger.generate_output_file_name(model.tokenizer.name_or_path, num_shots, dataset,
                                                                   decode_method, num_iter, num_beams, top_p)
    experiment_message = experiment_logger.generate_experiment_message(decode_method, model.tokenizer.name_or_path,
                                                                       num_shots, dataset, num_iter, num_beams, top_p)

    with open('./results_with_logits/' + output_file_name, 'w', encoding='utf-8') as f:
        print(experiment_message)

        for i in range(num_iter):
            query_info = query_maker.generate_shots_and_question(num_shots)
            query_template = """Here is some example:
{shots}
Please solve the following question
Question:
{final_question}
{final_choices}
Correct answer:"""

            custom_query_info = query_maker.create_custom_query(query_info, query_template=query_template,
                                                                show_correct_answer=True)

            model_inputs = model.generate_input(custom_query_info['query'])
            output = model.generate_output(model_inputs, decode_method, max_new_tokens=5, num_beams=num_beams,
                                           top_p=top_p)
            generated_text = model.decode_output(output)

            f.write('Next:\n')
            f.write('Query:\n')
            f.write(custom_query_info['query'] + '\n')
            f.write('Generation: \n')
            f.write(generated_text[len(custom_query_info['query']):] + '\n')
            f.write('Ground Truth Answer: \n')
            f.write(custom_query_info['answer'] + '\n')
            f.write('logistics:\n')
            sorted_token_probs = model.extract_and_sort_token_probabilities(output)
            for element in sorted_token_probs[:5]:
                f.write(f"  {element[0]}: {element[1]:.4f}" + '\n')


def experiment_with_modified_query(model, num_shots, dataset, decode_method, num_iter, num_beams=0, top_p=0.0, k=0):
    acc = 0
    dataset_loader = DatasetLoader()
    experiment_logger = ExperimentLogger()

    data = dataset_loader.load_dataset(dataset)
    query_maker = Query(data, dataset)
    output_file_name = experiment_logger.generate_output_file_name(model.tokenizer.name_or_path, num_shots, dataset,
                                                                   decode_method, num_iter, num_beams, top_p, k=k)
    POE_Wrong = -0
    with open('./results_with_logits/' + output_file_name, 'w', encoding='utf-8') as f:
        for iteration in range(num_iter):
            query_info = query_maker.generate_shots_and_question(num_shots)
            query_template = """Eliminate wrong options from the given choices for the question:
{shots}
Please solve the following question
Question:
{final_question}
Options:
{final_choices}
Incorrect answer:"""
            custom_query_info = query_maker.create_custom_query(query_info, query_template, show_correct_answer=False)

            model_inputs = model.generate_input(custom_query_info['query'])
            output = model.generate_output(model_inputs, decode_method, max_new_tokens=5, num_beams=num_beams,
                                           top_p=top_p)
            generated_text = model.decode_output(output)

            sorted_token_probs = model.extract_and_sort_token_probabilities(output)

            modified_query_info = modify_query_info_based_on_tokens(query_info, sorted_token_probs, k)
            query_template = """Choose a correct options from the given choices for the question:
{shots}
Please solve the following question
Question:
{final_question}
Options:
{final_choices}
Correct answer:"""
            modified_custom_query_info = query_maker.create_custom_query(modified_query_info, query_template,
                                                                         show_correct_answer=True)

            modified_model_inputs = model.generate_input(modified_custom_query_info['query'])
            modified_output = model.generate_output(modified_model_inputs, decode_method, max_new_tokens=5,
                                                    num_beams=num_beams, top_p=top_p)
            modified_sorted_token_probs = model.extract_and_sort_token_probabilities(modified_output)

            modified_generated_text = model.decode_output(modified_output)
            best_choices = [choice.replace("▁", "") for choice, _ in modified_sorted_token_probs[:1]]

            if modified_custom_query_info['answer'] == best_choices[0]:
                acc += 1
            if modified_custom_query_info['answer'] == 'Z':
                POE_Wrong += 1
            f.write(f"Iteration {iteration + 1}:\n")
            f.write('Initial Query:\n')
            f.write(custom_query_info['query'] + '\n')
            f.write('Initial Generation: \n')
            f.write(generated_text[len(custom_query_info['query']):] + '\n')
            f.write('Initial logit: \n')
            for element in sorted_token_probs[:5]:
                f.write(f"  {element[0]}: {element[1]:.4f}" + '\n')
            f.write('IGround Truth Answer: \n')
            f.write(custom_query_info['answer'] + '\n')
            f.write('Modified Query:\n')
            f.write(modified_custom_query_info['query'] + '\n')
            f.write('Modified Generation: \n')
            f.write(modified_generated_text[len(modified_custom_query_info['query']):] + '\n')
            f.write('Modified logit: \n')
            for element in modified_sorted_token_probs[:5]:
                f.write(f"  {element[0]}: {element[1]:.4f}" + '\n')
            f.write('Ground Truth Answer: \n')
            f.write(modified_custom_query_info['answer'] + '\n')
            f.write('-' * 20 + '\n\n')
        f.write('Acc is ' + str(acc / num_iter) + '\n')
        f.write('POE Wrong is ' + str(POE_Wrong / num_iter))


def experiment_with_test(model, num_shots, dataset, decode_method, num_iter, num_beams=0, top_p=0.0, POE=1,
                         Rate=90):
    acc = 0
    dataset_loader = DatasetLoader()
    experiment_logger = ExperimentLogger()

    data = dataset_loader.load_dataset(dataset)
    query_maker = Query(data, dataset)
    output_file_name = experiment_logger.generate_output_file_name(model.tokenizer.name_or_path, num_shots, dataset,
                                                                   decode_method, num_iter, num_beams, top_p, POE, Rate)
    start = time.time()
    with open('./results_with_logits/' + output_file_name, 'w', encoding='utf-8') as f:
        for iteration in range(num_iter):
            query_info = query_maker.generate_shots_and_question(num_shots)
            query_template = """Eliminate wrong options from the given choices for the question:
{shots}
Please solve the following question
Question:
{final_question}
Options:
{final_choices}
Incorrect answer:"""
            custom_query_info = query_maker.create_custom_query(query_info, query_template, show_correct_answer=False)

            model_inputs = model.generate_input(custom_query_info['query'])
            output = model.generate_output(model_inputs, decode_method, max_new_tokens=5, num_beams=num_beams,
                                           top_p=top_p)
            generated_text = model.decode_output(output)

            sorted_token_probs = model.extract_and_sort_token_probabilities(output)

            f.write(f"Iteration {iteration + 1}:\n")
            f.write('Initial Query:\n')
            f.write(custom_query_info['query'] + '\n')
            f.write('Initial Generation: \n')
            f.write(generated_text[len(custom_query_info['query']):] + '\n')
            f.write('Initial logit: \n')
            for element in sorted_token_probs[:5]:
                f.write(f"  {element[0]}: {element[1]:.4f}" + '\n')
            f.write('Ground Truth Answer: \n')
            f.write(custom_query_info['answer'] + '\n')
            f.write('-' * 20 + '\n\n')
        end = time.time()
        f.write("time is " + str(end - start))


# experiment_with_test(model, num_shots=5, dataset='commonsenseQA', decode_method='greedy', num_iter=2000)
experiment_with_modified_query(model, num_shots=5, dataset='commonsenseQA', decode_method='greedy', num_iter=2000,
                               k=150)
experiment_with_modified_query(model, num_shots=5, dataset='commonsenseQA', decode_method='greedy', num_iter=2000,
                               k=155)
experiment_with_modified_query(model, num_shots=5, dataset='commonsenseQA', decode_method='greedy', num_iter=2000,
                               k=160)
experiment_with_modified_query(model, num_shots=5, dataset='commonsenseQA', decode_method='greedy', num_iter=2000,
                               k=165)
experiment_with_modified_query(model, num_shots=5, dataset='commonsenseQA', decode_method='greedy', num_iter=2000,
                               k=170)
experiment_with_modified_query(model, num_shots=5, dataset='commonsenseQA', decode_method='greedy', num_iter=2000,
                               k=175)
experiment_with_modified_query(model, num_shots=5, dataset='commonsenseQA', decode_method='greedy', num_iter=2000,
                               k=180)
# experiment_with_modified_query(model, num_shots=5, dataset='commonsenseQA', decode_method='greedy', num_iter=2000, POE=2,
#                                Rate=50)
# experiment_with_modified_query(model, num_shots=5, dataset='commonsenseQA', decode_method='greedy', num_iter=2000, POE=2,
#                                Rate=40)
# experiment_with_modified_query(model, num_shots=5, dataset='commonsenseQA', decode_method='greedy', num_iter=2000, POE=2,
#                                Rate=80)
# experiment_with_modified_query(model, num_shots=5, dataset='commonsenseQA', decode_method='greedy', num_iter=2000, POE=2,
#                                Rate=30)
# experiment_with_modified_query(model, num_shots=5, dataset='OpenBookQA', decode_method='greedy', num_iter=300, POE=1,
#                                Rate=80)
# experiment_with_modified_query(model, num_shots=5, dataset='qasc', decode_method='greedy', num_iter=300, POE=1,
#                                Rate=80)
# experiment_with_modified_query(model, num_shots=5, dataset='commonsenseQA', decode_method='greedy', num_iter=300, POE=1,
#                                Rate=70)
# experiment_with_modified_query(model, num_shots=5, dataset='OpenBookQA', decode_method='greedy', num_iter=300, POE=1,
#                                Rate=70)
# experiment_with_modified_query(model, num_shots=5, dataset='qasc', decode_method='greedy', num_iter=300, POE=1,
#                                Rate=70)
# experiment_with_modified_query(model, num_shots=5, dataset='commonsenseQA', decode_method='greedy', num_iter=300, POE=1,
#                                Rate=60)
# experiment_with_modified_query(model, num_shots=5, dataset='OpenBookQA', decode_method='greedy', num_iter=300, POE=1,
#                                Rate=60)
# experiment_with_modified_query(model, num_shots=5, dataset='qasc', decode_method='greedy', num_iter=300, POE=1,
#                                Rate=60)
# experiment_with_modified_query(model, num_shots=5, dataset='commonsenseQA', decode_method='greedy', num_iter=300, POE=1,
#                                Rate=-1)
# experiment_with_modified_query(model, num_shots=5, dataset='OpenBookQA', decode_method='greedy', num_iter=300, POE=1,
#                                Rate=-1)
# experiment_with_modified_query(model, num_shots=5, dataset='qasc', decode_method='greedy', num_iter=300, POE=1,
#                                Rate=-1)
# experiment(model, num_shots=5, dataset='commonsenseQA', decode_method='greedy', num_iter=200)
# experiment(model, num_shots=5, dataset='qasc', decode_method='greedy', num_iter=1000)
# experiment(model, num_shots=5, dataset='OpenBookQA', decode_method='greedy', num_iter=1000)
# experiment(model, num_shots=5, dataset='MMLU', decode_method='greedy', num_iter=1000)
