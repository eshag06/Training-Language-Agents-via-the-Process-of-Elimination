import numpy as np
import torch as torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import random
import torch
import argparse
from load_datasets import *
import time
import torch.nn.functional as F
import copy


def generate_output_file_name(model_name, num_shots, dataset, decode_method, num_iter, num_beams, top_p):
    model_name = model_name.split('/')[-1]

    if decode_method == 'greedy':
        return f'result_{model_name}_{num_shots}_{dataset}_{decode_method}_{num_iter}.txt'
    elif decode_method == 'beam':
        return f'result_{model_name}_{num_shots}_{dataset}_{decode_method}_{num_iter}_{num_beams}.txt'
    elif decode_method == 'nucleus':
        return f'result_{model_name}_{num_shots}_{dataset}_{decode_method}_{num_iter}_{top_p}.txt'


def experiment_query_text(num_shots: int, data):
    """
    Generates a query based on a subset of given data and provides information for further customization.

    :param num_shots: Number of shots to include in the query.
    :param data: The whole dataset.
    :return: A tuple containing the complete query, the correct answer of the last question, and a dict with information about the selected shots and the final question.
    """
    random_elements = random.sample(data, num_shots + 1)
    query_info = {'shots': random_elements[:-1], 'final_question': random_elements[-1]}

    query = 'Choose an answer of these questions and give an explanation for your answer.\n'
    for element in random_elements[:-1]:
        question = element['question']
        query += question['stem'] + '\n'
        for choice in question['choices']:
            query += choice['label'] + ': ' + choice['text'] + '\n'
        query += 'The correct answer is ' + element['answerKey'] + '\n\n'

    final_q = random_elements[-1]['question']
    query += final_q['stem'] + '\n'
    for choice in final_q['choices']:
        query += choice['label'] + ': ' + choice['text'] + '\n'
    query += 'The incorrect answer is'

    return query, random_elements[-1]['answerKey'], final_q['choices'], query_info


def custom_query(shots, final_question):
    """
    Generates a custom query based on specified shots and a final question.

    :param shots: A list of shot elements.
    :param final_question: The final question element.
    :return: A custom query string and the correct answer of the final question.
    """
    query = 'Choose an answer of these questions and give an explanation for your answer.\n'
    for element in shots:
        question = element['question']
        query += question['stem'] + '\n'
        for choice in question['choices']:
            query += choice['label'] + ': ' + choice['text'] + '\n'
        query += 'The correct answer is ' + element['answerKey'] + '\n\n'

    q = final_question['question']
    query += q['stem'] + '\n'
    for choice in q['choices']:
        query += choice['label'] + ': ' + choice['text'] + '\n'
    query += 'The correct answer is'

    return query, final_question['answerKey']


def generate_experiment_message(decode_method, model_name, num_shots, dataset, num_iter, num_beams=None, top_p=None):
    """
    Generates and prints a message based on the decode method, model name, number of shots, dataset, and iteration number.
    Additional parameters for beam width and top_p are used if relevant.

    :param decode_method: The method used for decoding ('greedy', 'beam', or 'nucleus').
    :param model_name: Name of the model being used.
    :param num_shots: Number of shots used in the query.
    :param dataset: Name of the dataset being used.
    :param num_iter: The iteration number of the query being generated.
    :param num_beams: The number of beams used in beam search decoding (optional).
    :param top_p: The nucleus sampling probability (optional).
    """
    if decode_method == 'greedy':
        message = f'Generating queries with {model_name}, {num_shots}-shot with dataset {dataset}, using {decode_method} method to decode.\n'
    elif decode_method == 'beam':
        if num_beams is None:
            raise ValueError("num_beams must be specified for beam search decoding.")
        message = f'Generating queries with {model_name}, {num_shots}-shot with dataset {dataset}, using {decode_method} method to decode, beam width is {num_beams}.\n'
    elif decode_method == 'nucleus':
        if top_p is None:
            raise ValueError("top_p must be specified for nucleus sampling decoding.")
        message = f'Generating queries with {model_name}, {num_shots}-shot with dataset {dataset}, using {decode_method} method to decode, top p is {top_p}.\n '
    else:
        raise ValueError("Invalid decode_method. Choose from 'greedy', 'beam', or 'nucleus'.")

    message += f'Query number: {num_iter} \n'
    print(message)


def generate_output(model, model_inputs, decode_method, max_new_tokens, num_beams=None, top_p=None):
    """
    Generate model output using specified decoding method.

    Args:
    - model: The model to generate outputs from.
    - model_inputs: The inputs to the model.
    - decode_method: The decoding strategy ('greedy', 'beam', or 'nucleus').
    - max_new_tokens: The maximum number of new tokens to generate.
    - num_beams: The number of beams for beam search (required if decode_method is 'beam').
    - top_p: The nucleus sampling probability (required if decode_method is 'nucleus').

    Returns:
    - output: The generated output from the model.
    """

    if decode_method == 'greedy':
        output = model.generate(
            **model_inputs,
            temperature=0.8,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True
        )
    elif decode_method == 'beam':
        if num_beams is None:
            raise ValueError("num_beams must be specified for beam search decoding.")
        output = model.generate(
            **model_inputs,
            temperature=0.8,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True
        )
    elif decode_method == 'nucleus':
        if top_p is None:
            raise ValueError("top_p must be specified for nucleus sampling decoding.")
        output = model.generate(
            **model_inputs,
            temperature=0.8,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            top_k=0,
            return_dict_in_generate=True,
            output_scores=True
        )
    else:
        raise ValueError("Invalid decode_method. Choose from 'greedy', 'beam', or 'nucleus'.")

    return output


def make_query(model_name, num_shots, dataset, decode_method, num_iter, num_beams=0, top_p=0.0):
    generate_experiment_message(decode_method, model_name, num_shots, dataset, num_iter, num_beams=num_beams,
                                top_p=top_p)
    max_new_tokens = 5
    seconds = 0
    if dataset == 'commonsenseQA':
        data = load_commonsenseQA()
    elif dataset == 'reducedcommonsenseQA':
        data = load_commonsenseQA('reduce')
    elif dataset == 'OpenBookQA':
        data = load_commonsenseQA()

    output_file_name = generate_output_file_name(model_name, num_shots, dataset, decode_method, num_iter, num_beams,
                                                 top_p)
    with open('./results_with_logits/' + output_file_name, 'w', encoding='utf-8') as f:
        for i in range(num_iter):
            query, answer, choice, query_info = experiment_query_text(num_shots, data)
            # Encode input text and generate output
            model_inputs = tokenizer(query, return_tensors='pt').to(torch_device)

            start_time = time.time()
            with torch.no_grad():
                # Generate text - adjust parameters like max_length as needed
                output = generate_output(model, model_inputs, decode_method, max_new_tokens, num_beams=None, top_p=None)
                generated_text = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
                probabilities = [torch.softmax(logit, dim=-1) for logit in output.scores]
                # Convert all token IDs to tokens for matching
                vocab_tokens = tokenizer.convert_ids_to_tokens(range(model.config.vocab_size))

                for step, step_probs in enumerate(probabilities):
                    # For each step, match each token's probability
                    token_probs = zip(vocab_tokens, step_probs[0].tolist())
                    # Now you have a list of (token, probability) pairs for this step
                    # You can sort or filter this list as needed
                    sorted_token_probs = sorted(token_probs, key=lambda x: x[1], reverse=True)
                    break

            end_time = time.time()
            seconds += end_time - start_time

            f.write('Next:\n')
            f.write('Query:\n')
            f.write(query + '\n')
            f.write('Generation: \n')
            f.write(generated_text[len(query):] + '\n')
            f.write('Ground Truth Answer: \n')
            f.write(answer + '\n')
            f.write('Logits:\n')
            for token, prob in sorted_token_probs[:5]:
                f.write(f"  {token}: {prob:.4f}" + '\n')
        f.write(f'Average decoding time is {seconds / num_iter} seconds')


def make_query_POE(model_name, num_shots, dataset, decode_method, num_iter, num_beams=0, top_p=0.0):
    generate_experiment_message(decode_method, model_name, num_shots, dataset, num_iter, num_beams=num_beams,
                                top_p=top_p)
    max_new_tokens = 5
    seconds = 0

    if dataset == 'commonsenseQA':
        data = load_commonsenseQA()
        data = copy.deepcopy(data)
    elif dataset == 'reducedcommonsenseQA':
        data = load_commonsenseQA('reduce')
    elif dataset == 'OpenBookQA':
        data = load_commonsenseQA()

    output_file_name = generate_output_file_name(model_name, num_shots, dataset, decode_method, num_iter, num_beams,
                                                 top_p)
    with open('./results_with_logits/' + output_file_name, 'w', encoding='utf-8') as f:
        for i in range(num_iter):
            query, answer, choices, query_info = experiment_query_text(num_shots, copy.deepcopy(data))
            print(choices)
            model_inputs = tokenizer(query, return_tensors='pt').to(torch_device)
            start_time = time.time()
            with torch.no_grad():
                output = generate_output(model, model_inputs, decode_method, max_new_tokens, num_beams=num_beams,
                                         top_p=top_p)
                generated_text = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
                probabilities = [torch.softmax(logit, dim=-1) for logit in output.scores]
                vocab_tokens = tokenizer.convert_ids_to_tokens(range(model.config.vocab_size))

                for step, step_probs in enumerate(probabilities):
                    token_probs = zip(vocab_tokens, step_probs[0].tolist())
                    sorted_token_probs = sorted(token_probs, key=lambda x: x[1], reverse=True)
                    break

            removed_choices = [choice.replace("▁", "") for choice, _ in sorted_token_probs[:2]]
            new_choices = [choice for choice in choices[:] if
                           choice['label'] not in removed_choices]
            dict = {}
            for i, choice in enumerate(new_choices):
                dict[choice['label']] = chr(65 + i)
                choice['label'] = chr(65 + i)
            if answer in removed_choices:
                updated_answer = 'F'
            else:
                print(dict)
                print(answer)
                print(choices)
                updated_answer = dict[answer]
            query_info['final_question']['question']['choices'] = new_choices
            custom_query_text, _ = custom_query(query_info['shots'],
                                                query_info['final_question'])
            model_inputs = tokenizer(custom_query_text, return_tensors='pt').to(torch_device)
            with torch.no_grad():
                output = generate_output(model, model_inputs, decode_method, max_new_tokens, num_beams=num_beams,
                                         top_p=top_p)
                second_generated_text = tokenizer.decode(output.sequences[0], skip_special_tokens=True)

            end_time = time.time()
            seconds += end_time - start_time
            f.write('Next:\n')
            f.write('Query:\n')
            f.write(custom_query_text + '\n')
            f.write('Generation: \n')
            f.write(second_generated_text[len(custom_query_text):] + '\n')
            f.write('Ground Truth Answer: \n')
            f.write(updated_answer + '\n')
            f.write('Logits:\n')
            for token, prob in sorted_token_probs[:5]:
                f.write(f"  {token}: {prob:.4f}" + '\n')
        f.write(f'Average decoding time is {seconds / num_iter} seconds\n')


if __name__ == '__main__':
    # Replace 'model-name' with the appropriate model name for LLaMA-2
    model_name = 'meta-llama/Llama-2-7b-hf'
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model - make sure to use the correct model class for text generation (e.g., AutoModelForCausalLM)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(torch_device)
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # breakpoint()
    # num_shots = 2
    num_iter = 1000
    for num_shots in [2, 5]:
        make_query_POE(model_name, num_shots, 'commonsenseQA', 'greedy', num_iter)
        for num_beams in [2, 5]:
            make_query_POE(model_name, num_shots, 'commonsenseQA', 'beam', num_iter, num_beams=num_beams)
        for top_p in [0.7, 0.8, 0.9]:
            make_query_POE(model_name, num_shots, 'commonsenseQA', 'nucleus', num_iter, top_p=top_p)

