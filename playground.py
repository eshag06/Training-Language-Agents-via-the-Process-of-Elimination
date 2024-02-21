import numpy as np
from torch import Tensor
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import random
import torch
import argparse
from load_datasets import *
import time
import torch.nn.functional as F

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

    :param n: Number of shots.
    :param data: The whole dataset.
    :return: Complete query and the correct answer of the last question.
    """

    random_elements = random.sample(data, num_shots+1)
    query = 'Choose an answer of these questions and give an explanation for your answer.\n'
    for question in random_elements[:-1]:
        q = question['question']
        query += q['stem'] + '\n'
        for choice in q['choices']:
            query += choice['label'] + ': ' + choice['text'] + '\n'
        query += 'The correct answer is ' + question['answerKey'] + '\n\n'

    q = random_elements[-1]['question']
    query += q['stem'] + '\n'
    for choice in q['choices']:
        query += choice['label'] + ': ' + choice['text'] + '\n'
    query += 'The correct answer is'

    return query, random_elements[-1]['answerKey']


def make_query(model_name, num_shots, dataset, decode_method, num_iter, num_beams=0, top_p=0.0):
    if decode_method == 'greedy':
        print(f'Generating queries with {model_name}, {num_shots}-shot with dataset {dataset}, using {decode_method} method to decode.\n')
    elif decode_method == 'beam':
        print(
            f'Generating queries with {model_name}, {num_shots}-shot with dataset {dataset}, using {decode_method} method to decode, beam width is {num_beams}.\n')
    elif decode_method == 'nucleus':
        print(
            f'Generating queries with {model_name}, {num_shots}-shot with dataset {dataset}, using {decode_method} method to decode, top p is {top_p}.\n')
    print(f'Query number: {num_iter} \n')

    max_new_tokens = 2
    seconds = 0

    if dataset == 'commonsenseQA':
        data = load_commonsenseQA()
    elif dataset == 'reducedcommonsenseQA':
        data = load_commonsenseQA('reduce')

    output_file_name = generate_output_file_name(model_name, num_shots, dataset, decode_method, num_iter, num_beams, top_p)
    with open('./results_with_logits/' + output_file_name, 'w', encoding='utf-8') as f:
        for i in range(num_iter):
            input_text, answer = experiment_query_text(num_shots, data)
            # Encode input text and generate output
            model_inputs = tokenizer(input_text, return_tensors='pt').to(torch_device)

            start_time = time.time()
            with torch.no_grad():
                # Generate text - adjust parameters like max_length as needed
                if decode_method == 'greedy':
                    output = model.generate(
                        **model_inputs,
                        temperature=0.8,
                        max_new_tokens=max_new_tokens,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                elif decode_method == 'beam':
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

                input_length = 1 if model.config.is_encoder_decoder else model_inputs.input_ids.shape[1]
                generated_tokens = output.sequences[:, input_length:]
                generated_text = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
                transition_scores = model.compute_transition_scores(
                    output.sequences, output.scores, normalize_logits=True
                )
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
            f.write(input_text + '\n')
            f.write('Generation: \n')
            f.write(generated_text[len(input_text):] + '\n')
            f.write('Ground Truth Answer: \n')
            f.write(answer + '\n')
            f.write('Logits:\n')
            for token, prob in sorted_token_probs[:5]:
                f.write(f"  {token}: {prob:.4f}" +  '\n')

        f.write(f'Average decoding time is {seconds / num_iter} seconds')


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
    commonsenseQA = load_commonsenseQA()
    reduced_commonsenseQA = load_commonsenseQA('reduce')

    num_iter = 1000
    # make_query(model_name, 2, 'commonsenseQA', 'greedy', num_iter)

    # for num_shots in [2, 5]:
    #     make_query(model_name, num_shots, 'commonsenseQA', 'greedy', num_iter)
    #     for num_beams in [2, 3, 4, 5]:
    #         make_query(model_name, num_shots, 'commonsenseQA', 'beam', num_iter, num_beams=num_beams)
    #     for top_p in [0.7, 0.75, 0.8, 0.85, 0.9, 0.92]:
    #         make_query(model_name, num_shots, 'commonsenseQA', 'nucleus', num_iter, top_p=top_p)

    for num_shots in [2, 5]:
        make_query(model_name, num_shots, 'reducedcommonsenseQA', 'greedy', num_iter)
        for num_beams in [2, 3, 4, 5]:
            make_query(model_name, num_shots, 'reducedcommonsenseQA', 'beam', num_iter, num_beams=num_beams)
        for top_p in [0.7, 0.75, 0.8, 0.85, 0.9, 0.92]:
            make_query(model_name, num_shots, 'reducedcommonsenseQA', 'nucleus', num_iter, top_p=top_p)

