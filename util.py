import copy
import json
import random
import csv
import re

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class DatasetLoader:
    def __init__(self):
        # Initialize dataset paths for supported datasets.
        self.dataset_paths = {
            'commonsenseQA': "datasets/train_rand_split.jsonl",
            'MMLU': "datasets/MMLU.jsonl",
            'qasc': "datasets/qasc_dataset.jsonl",
            'OpenBookQA': 'datasets/train.jsonl',
            'Arc_easy': 'datasets/ARC-Easy-Train.jsonl',
            'Arc_hard': 'datasets/ARC-Challenge-Train.jsonl',
        }

    def load_dataset(self, dataset_name, modify_num_answer='', n=2):
        # Load a dataset by name. Optionally modify the number of answer choices.
        if dataset_name not in self.dataset_paths:
            raise ValueError(f"Dataset {dataset_name} is not supported.")  # Validate dataset name.
        # Get file path for the dataset.
        file_path = self.dataset_paths[dataset_name]
        # Load JSONL for other datasets.
        data = self.load_jsonl(file_path)
        if modify_num_answer == 'reduce':
            # Modify number of answer choices if requested.
            data = self.reduce_randomly_and_relabel_choices(data, n)

        return data

    @staticmethod
    def load_jsonl(file_path):
        # Load data from a JSONL file.
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                json_object = json.loads(line)
                data.append(json_object)
        return data


    @staticmethod
    def experiment_query_text(num_shots: int, data):
        """
        Generate a query for an experiment based on a subset of the data.

        :param num_shots: Number of examples to include in the query.
        :param data: The dataset from which to sample.
        :return: A dict with info about the selected examples and the final question for the experiment.
        """
        random_elements = random.sample(data, num_shots + 1)  # Randomly sample data for the query.
        query_info = {'shots': random_elements[:-1], 'final_question': random_elements[-1]}  # Organize query info.
        return query_info

    @staticmethod
    def reduce_randomly_and_relabel_choices(data, n):
        # Reduce the number of answer choices randomly and relabel the choices.
        for item in data:
            # Find the correct choice.
            correct_choice = next(
                (choice for choice in item['question']['choices'] if choice['label'] == item['answerKey']), None)
            if correct_choice is None:  # Skip if no correct choice is found.
                continue

            # Select incorrect choices and shuffle them.
            incorrect_choices = [choice for choice in item['question']['choices'] if
                                 choice['label'] != item['answerKey']]
            random.shuffle(incorrect_choices)
            selected_incorrect_choices = incorrect_choices[:n]  # Select N incorrect choices.

            # Combine selected choices and sort by text for consistency.
            combined_choices = [correct_choice] + selected_incorrect_choices
            combined_choices.sort(key=lambda x: x['text'])

            # Relabel choices and update the answer key.
            for i, choice in enumerate(combined_choices):
                new_label = chr(65 + i)  # Relabel starting with 'A'.
                if choice['text'] == correct_choice['text']:
                    item['answerKey'] = new_label  # Update answer key.
                choice['label'] = new_label  # Update choice label.

            item['question']['choices'] = combined_choices  # Update item choices.
        return data


class Query:
    def __init__(self, data, dataset_name):
        """
        Initialize the Query object with a dataset and automatically determine the data type based on the dataset name.

        :param data: The dataset to use for generating queries.
        :param dataset_name: The name of the dataset which determines the data type.
        """
        self.data = data
        self.dataset_name = dataset_name.lower()
        # Determine data type based on the dataset name
        self.data_type = 'json'

    def shuffle_choices_and_update_answer_randomly(self, data):
        """
        Shuffle the choices in a question randomly and update the answer key accordingly.

        :param seed:
        :param data: A dictionary containing the original question data, including the answer key and choices.
        :return: A dictionary with the choices shuffled and the answer key updated to reflect the new correct choice.
        """
        choices = data["question"]["choices"]
        answer_key = data["answerKey"]

        # Find the correct answer choice for later comparison
        correct_choice_text = next(choice["text"] for choice in choices if choice["label"] == answer_key)
        # Shuffle the choices
        random.shuffle(choices)

        # Assign new labels and find the new answer key
        for i, choice in enumerate(choices):
            choice["label"] = chr(65 + i)  # Update the label based on new order
            if choice["text"] == correct_choice_text:
                new_answer_key = choice["label"]

        # Update the data
        updated_data = data
        updated_data["answerKey"] = new_answer_key
        updated_data["question"]["choices"] = choices

        return updated_data

    def generate_shots_and_question(self, num_shots: int):
        """
        Generate shots and a final question from the dataset.

        :param seed:
        :param num_shots: Number of shots to include.
        :return: A dictionary containing the shots, the final question, and additional info.
        """
        # Select random elements from the dataset
        random_elements = random.sample(copy.deepcopy(self.data), num_shots + 1)
        for i in range(num_shots + 1):
            random_elements[i] = self.shuffle_choices_and_update_answer_randomly(random_elements[i])
        query_info = {
            'shots': random_elements[:-1],
            'final_question': random_elements[-1]
        }
        return query_info

    def create_custom_query(self, query_info, query_template, show_correct_answer=True):
        """
        Create a custom query text based on provided shots, a final question, and a query template,
        showing all choices but emphasizing either the correct answer or all incorrect answers based on the show_correct_answer flag,
        with only the answer labels (e.g., "A", "B", "C", "D") shown in the emphasized answer.

        :param query_info: A dictionary containing shots, the final question, and additional info.
        :param query_template: A string template for formatting the custom query.
        :param show_correct_answer: A boolean flag to emphasize the correct answer (True) or all incorrect answers (False).
        :return: A dictionary representing the custom query, including the query text, the correct answer label for the final question, and additional info.
        """
        output = {}
        query = query_template
        shots_text = ""
        previously_selected_incorrect_labels = []
        for element in query_info['shots']:
            if self.data_type == 'json':
                question = element['question']['stem']
                choices = element['question']['choices']
                choices_text = "\n".join([f"{choice['label']}: {choice['text']}" for choice in choices])
                correct_choice_label = element['answerKey']
                incorrect_choice_labels = [choice['label'] for choice in choices if
                                           choice['label'] != element['answerKey']]
                random.shuffle(incorrect_choice_labels)
                selected_incorrect_choice = next((choice for choice in incorrect_choice_labels if
                                                  choice not in previously_selected_incorrect_labels), None)
                if not selected_incorrect_choice:
                    selected_incorrect_choice = random.choice(incorrect_choice_labels)
                previously_selected_incorrect_labels.append(selected_incorrect_choice)
                emphasized_answer = f"Correct Answer: {correct_choice_label}" if show_correct_answer else f"Incorrect Answers: {', '.join(selected_incorrect_choice)}"

            shots_text += f"Questions:\n{question}\nOptions:\n{choices_text}\n{emphasized_answer}\n\n"

        final_question = query_info['final_question']
        final_q_text = final_question['question']['stem'] if self.data_type == 'json' else final_question[0]
        final_choices_text = "\n".join(
            [f"{choice['label']}: {choice['text']}" for choice in final_question['question']['choices']]
        ) if self.data_type == 'json' else "\n".join([f"{chr(65 + i)}: {final_question[1 + i]}" for i in range(4)])

        query = query.format(shots=shots_text, final_question=final_q_text, final_choices=final_choices_text)
        output['query'] = query
        output['answer'] = final_question['answerKey'] if self.data_type == 'json' else final_question[5]
        output['query_info'] = query_info
        return output


class Model:
    def __init__(self, model_name='meta-llama/Llama-2-7b-hf'):
        """
        Initialize the model and tokenizer for text generation.

        Args:
        - model_name: The name of the model to load.
        """
        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the model - make sure to use the correct model class for text generation
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.torch_device)
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_input(self, query):
        model_inputs = self.tokenizer(query, return_tensors='pt').to(self.torch_device)
        return model_inputs

    def generate_output(self, model_inputs, decode_method, max_new_tokens, num_beams=None, top_p=None):
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
        with torch.no_grad():
            if decode_method == 'greedy':
                output = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            elif decode_method == 'beam':
                if num_beams is None:
                    raise ValueError("num_beams must be specified for beam search decoding.")
                output = self.model.generate(
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
                output = self.model.generate(
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

    def decode_output(self, output):
        generated_text = self.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        return generated_text

    def extract_and_sort_token_probabilities(self, output):
        """
        Extracts generated text and sorted token probabilities from the model output.

        Parameters:
        - output: The output from the model after prediction.
        - tokenizer: The tokenizer used for encoding and decoding texts.
        - model: The language model used for prediction.

        Returns:
        - sorted_token_probs: A list where each element is a sorted list of (token, probability) pairs for each decoding step.
        """
        probabilities = [logit for logit in output.scores]
        vocab_tokens = self.tokenizer.convert_ids_to_tokens(range(self.model.config.vocab_size))

        sorted_token_probs = []

        for step, step_probs in enumerate(probabilities):
            # For each step, match each token's probability
            token_probs = zip(vocab_tokens, step_probs[0].tolist())
            # Now you have a list of (token, probability) pairs for this step
            # You can sort or filter this list as needed
            filtered_token_probs = filter(lambda x: re.match(r"^▁[A-Z]", x[0]), token_probs)
            # Now you have a filtered list of (token, probability) pairs for this step
            # Sort this filtered list
            sorted_token_probs = sorted(filtered_token_probs, key=lambda x: x[1], reverse=True)
            break  # Assuming you only want to process the first step here

        return sorted_token_probs[:10]


class ExperimentLogger:
    def __init__(self):
        """
        Initializes the ExperimentLogger with a model, tokenizer, and device.

        :param model: The pre-trained model from Hugging Face's Transformers library.
        :param tokenizer: The tokenizer corresponding to the pre-trained model.
        :param torch_device: The torch device (e.g., 'cpu', 'cuda') on which computations will be performed.
        """

    def generate_output_file_name(self, model_name, num_shots, dataset, decode_method, num_iter, num_beams=None,
                                  top_p=None, POE=None, Rate=None, k=None, n=None, seed=None):
        """
        Generates a file name for the experiment's output based on various parameters.

        :returns: The generated file name.
        """
        model_name = model_name.split('/')[-1]
        if decode_method == 'greedy':
            return f'result_{model_name}_{num_shots}_{dataset}_{decode_method}_{num_iter}_{seed}_{n}_{k}.txt'
        elif decode_method == 'beam':
            return f'result_{model_name}_{num_shots}_{dataset}_{decode_method}_{num_iter}_{num_beams}.txt'

        elif decode_method == 'nucleus':
            return f'result_{model_name}_{num_shots}_{dataset}_{decode_method}_{num_iter}_{top_p}.txt'

    def generate_experiment_message(self, decode_method, model_name, num_shots, dataset, num_iter, num_beams=None,
                                    top_p=None):
        """
        Generates a message based on the experiment setup.

        :returns: The generated message as a string.
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
            message = f'Generating queries with {model_name}, {num_shots}-shot with dataset {dataset}, using {decode_method} method to decode, top p is {top_p}.\n'
        else:
            raise ValueError("Invalid decode_method. Choose from 'greedy', 'beam', or 'nucleus'.")

        message += f'Query number: {num_iter}\n'


class POE_threshold:
    def __init__(self):
        """
        method1
        """

    def modify_query_info_based_on_tokens(self, query_info, sorted_token_probs, k):
        probs = [prob for _, prob in sorted_token_probs if prob >= 0]
        average_prob = np.mean(probs)
        std_dev = np.std(probs)
        choices = query_info['final_question']["question"]["choices"]
        answer = query_info['final_question']['answerKey']
        correct_choice_text = next(choice["text"] for choice in choices if choice["label"] == answer)

        threshold = 1 * average_prob + k / 100 * std_dev
        removed_choices = [choice.replace("▁", "") for choice, prob in sorted_token_probs if prob > threshold]
        if len(removed_choices) != 0:
            removed_choices = removed_choices
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

    def experiment_with_modified_query(self, model, num_shots, dataset, decode_method, num_iter, num_beams=0, top_p=0.0,
                                       k=0,
                                       n=0, seed=123):
        acc = 0
        dataset_loader = DatasetLoader()
        experiment_logger = ExperimentLogger()

        data = dataset_loader.load_dataset(dataset)
        if n != 0:
            data = dataset_loader.reduce_randomly_and_relabel_choices(data, n)
        query_maker = Query(data, dataset)
        output_file_name = experiment_logger.generate_output_file_name(model.tokenizer.name_or_path, num_shots, dataset,
                                                                       decode_method, num_iter, num_beams, top_p,
                                                                       seed=seed,
                                                                       k=k, n=n)
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
                custom_query_info = query_maker.create_custom_query(query_info, query_template,
                                                                    show_correct_answer=False)

                model_inputs = model.generate_input(custom_query_info['query'])
                output = model.generate_output(model_inputs, decode_method, max_new_tokens=5, num_beams=num_beams,
                                               top_p=top_p)
                generated_text = model.decode_output(output)

                sorted_token_probs = model.extract_and_sort_token_probabilities(output)

                modified_query_info = self.modify_query_info_based_on_tokens(query_info, sorted_token_probs, k)
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


class tournament_style:
    def __init__(self):
        """
        method2
        """

    def model_response(self, model_answer):
        match = re.search(r'[A-Z]', model_answer)  # This regex pattern looks for uppercase letters
        if match:
            extracted_char = match.group()
            print(extracted_char)
        else:
            return []
        incorrect_labels = []
        if extracted_char in ['A', 'B', 'C', 'D', 'E', 'F']:  # Assuming choices are labeled from A to E
            incorrect_labels.append(extracted_char)

        return incorrect_labels

    def experiment_tournament_style(self, model, dataset, num_remaining_choices, name, write=True):
        DL = DatasetLoader()
        data = DL.load_dataset(dataset)
        examples = data[-100:]
        pc = []

        write_content = ""

        with open(f'./round/{name}.txt', 'w', encoding='utf-8') as f:
            for d in data[:4000]:
                write_content += '\nNext:\n'
                write_content += '\nQuery:\n'

                num_round = 1
                possible_choices = d['question']['choices'][:]
                while len(possible_choices) > 1:
                    random.shuffle(possible_choices)
                    example = random.sample(examples, 2)
                    query = ''
                    write_content += f'\nRound {num_round}\'s query:\n'
                    prompt_gt = []
                    query += '### Instructions: Let\'s think step by step, pick the answer that is more reasonable.\n'
                    for e in example:
                        temp_query = ''
                        temp_query += '### Question:\n'
                        temp_query += e['question']['stem'] + '\n'
                        temp_query += '### Options:\n'
                        ic = [c for c in e['question']['choices'] if c['label'] != e['answerKey']]
                        gt = [c for c in e['question']['choices'] if c['label'] == e['answerKey']][0]
                        random.shuffle(ic)
                        ic = random.sample(ic, 1)[0]
                        if len(prompt_gt) == 0:
                            gt_l = chr(random.sample([65, 66], 1)[0])
                        else:
                            gt_l = chr(131 - ord(prompt_gt[0]))
                            prompt_gt.clear()
                        ic_l = chr(131 - ord(gt_l))
                        prompt_gt.append(gt_l)
                        if gt_l == 'A':
                            temp_query += f"{gt_l}: {gt['text']}\n"
                            temp_query += f"{ic_l}: {ic['text']}\n"
                        else:
                            temp_query += f"{ic_l}: {ic['text']}\n"
                            temp_query += f"{gt_l}: {gt['text']}\n"
                        query += temp_query

                        query += f'\nThe better answer is: {gt_l}\n\n'

                    query += '### Question:\n'
                    query += d['question']['stem'] + '\n'

                    t = []
                    query += '### Options:\n'
                    for i, choice in enumerate(random.sample(possible_choices, 2)):
                        query += f"{chr(i + 65)}: {choice['text']}\n"
                        t.append(choice['text'])

                    query += 'The better answer is:'

                    write_content += query + '\n'

                    model_inputs = model.generate_input(query)
                    output = model.generate_output(model_inputs, "greedy", max_new_tokens=5)
                    generated_text = model.decode_output(output)

                    sorted_token_probs = model.extract_and_sort_token_probabilities(output)

                    token, prob = sorted_token_probs[0]
                    generated_text = token
                    write_content += f'\nRound {num_round}\'s respond:\n'
                    write_content += generated_text + '\n'

                    if 'I think they are equally good.' in generated_text:
                        pass
                    else:
                        choices = self.model_response(generated_text)
                        # choices = []
                        if len(choices) == 0:
                            pass
                        else:
                            correct_choice = choices[0]
                            if correct_choice == 'B':
                                for c in possible_choices:
                                    if c['text'] == t[0]:
                                        possible_choices.remove(c)
                                        break
                            elif correct_choice == 'A':
                                for c in possible_choices:
                                    if c['text'] == t[1]:
                                        possible_choices.remove(c)
                                        break
                            if len(possible_choices) <= num_remaining_choices:
                                pc.append(possible_choices)
                                continue

                    num_round += 1
                    if len(possible_choices) == 1:
                        correct_answer = possible_choices[0]['label']
                        write_content += f'\n Correct answer: {correct_answer}\n'
                        break

                    if num_round > 10:
                        r = [c['label'] for c in possible_choices]
                        write_content += f'\nRemaining choices:{r}\n'
                        break

                write_content += '\nGround truth answer:\n'
                write_content += d['answerKey']
        if write:
            f.write(write_content)
        return pc
