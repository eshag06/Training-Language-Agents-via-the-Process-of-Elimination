import copy
import json
import random
import csv

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class DatasetLoader:
    def __init__(self):
        # Initialize dataset paths for supported datasets.
        self.dataset_paths = {
            'commonsenseQA': "datasets/train_rand_split.jsonl",
            'MMLU': "datasets/mc_test.csv",
            'qasc': "datasets/qasc_dataset.jsonl",
            'OpenBookQA': 'datasets/train.jsonl'
        }

    def load_dataset(self, dataset_name, modify_num_answer='', n=2):
        # Load a dataset by name. Optionally modify the number of answer choices.
        if dataset_name not in self.dataset_paths:
            raise ValueError(f"Dataset {dataset_name} is not supported.")  # Validate dataset name.
        # Get file path for the dataset.
        file_path = self.dataset_paths[dataset_name]

        if dataset_name == 'MMLU':
            # Load CSV for MMLU dataset.
            data = self.load_csv(file_path)
        else:
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
    def load_csv(file_path):
        # Load data from a CSV file.
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            csvreader = csv.reader(file, delimiter=',')
            for row in csvreader:
                data.append(row)
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
        self.data_type = 'csv' if dataset_name.lower() in ['mmlu'] else 'json'

    def generate_shots_and_question(self, num_shots: int):
        """
        Generate shots and a final question from the dataset.

        :param num_shots: Number of shots to include.
        :return: A dictionary containing the shots, the final question, and additional info.
        """
        # Select random elements from the dataset
        random_elements = random.sample(copy.deepcopy(self.data), num_shots + 1)
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
        for element in query_info['shots']:
            if self.data_type == 'json':
                question = element['question']['stem']
                choices = element['question']['choices']
                choices_text = "\n".join([f"{choice['label']}: {choice['text']}" for choice in choices])
                correct_choice_label = element['answerKey']
                incorrect_choice_labels = [choice['label'] for choice in choices if
                                           choice['label'] != element['answerKey']]
                emphasized_answer = f"Correct Answer: {correct_choice_label}" if show_correct_answer else f"Incorrect Answers: {', '.join(incorrect_choice_labels)}"
            elif self.data_type == 'csv':
                question = element[0]
                choices = element[1:5]
                choices_text = "\n".join([f"{chr(65 + i)}: {choice}" for i, choice in enumerate(choices)])
                correct_answer_label = element[5]
                incorrect_choice_labels = [chr(65 + i) for i, choice in enumerate(choices) if
                                           chr(65 + i) != correct_answer_label]
                emphasized_answer = f"Correct Answer: {correct_answer_label}" if show_correct_answer else f"Incorrect Answers: {', '.join(incorrect_choice_labels)}"

            shots_text += f"{question}\n{choices_text}\n{emphasized_answer}\n\n"

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

        if decode_method == 'greedy':
            output = self.model.generate(
                **model_inputs,
                temperature=0.8,
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
        probabilities = [torch.softmax(logit, dim=-1) for logit in output.scores]
        vocab_tokens = self.tokenizer.convert_ids_to_tokens(range(self.model.config.vocab_size))

        sorted_token_probs = []

        for step, step_probs in enumerate(probabilities):
            # For each step, match each token's probability
            token_probs = zip(vocab_tokens, step_probs[0].tolist())
            # Now you have a list of (token, probability) pairs for this step
            # You can sort or filter this list as needed
            sorted_token_probs = sorted(token_probs, key=lambda x: x[1], reverse=True)
            break

        return sorted_token_probs[:5]


class ExperimentLogger:
    def __init__(self):
        """
        Initializes the ExperimentLogger with a model, tokenizer, and device.

        :param model: The pre-trained model from Hugging Face's Transformers library.
        :param tokenizer: The tokenizer corresponding to the pre-trained model.
        :param torch_device: The torch device (e.g., 'cpu', 'cuda') on which computations will be performed.
        """

    def generate_output_file_name(self, model_name, num_shots, dataset, decode_method, num_iter, num_beams=None,
                                  top_p=None, POE=None):
        """
        Generates a file name for the experiment's output based on various parameters.

        :returns: The generated file name.
        """
        model_name = model_name.split('/')[-1]
        if decode_method == 'greedy':
            return f'result_{model_name}_{num_shots}_{dataset}_{decode_method}_{num_iter}.txt'
        elif decode_method == 'beam':
            if POE is not None:
                return f'result_{model_name}_{num_shots}_{dataset}_{decode_method}_{num_iter}_{num_beams}_{POE}.txt'
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
