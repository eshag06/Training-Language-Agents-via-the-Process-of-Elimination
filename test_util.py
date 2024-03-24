import copy

import torch

from util import DatasetLoader
from util import Query
from util import Model
from util import ExperimentLogger

model = Model(model_name='meta-llama/Llama-2-7b-hf')


def modify_query_info_based_on_tokens(query_info, sorted_token_probs, top_n):
    removed_choices = [choice.replace("▁", "") for choice, _ in sorted_token_probs[:top_n]]
    answer = query_info['final_question']['answerKey']
    new_choices = [choice for choice in query_info['final_question']['question']['choices'] if
                   choice['label'] not in removed_choices]
    dict = {}
    for i, choice in enumerate(new_choices):
        dict[choice['label']] = chr(65 + i)
        choice['label'] = chr(65 + i)
    if answer in removed_choices:
        updated_answer = 'Z'
    else:
        updated_answer = dict[answer]
    query_info['final_question']['question']['choices'] = new_choices
    query_info['answerKey'] = updated_answer

    return query_info


def experiment(model, num_shots, dataset, decode_method, num_iter, num_beams=0, top_p=0.0):
    dataset_loader = DatasetLoader()
    experiment_logger = ExperimentLogger()

    # 加载数据集
    data = dataset_loader.load_dataset(dataset)
    query_maker = Query(data, dataset)

    # 生成输出文件名和实验消息
    output_file_name = experiment_logger.generate_output_file_name(model.tokenizer.name_or_path, num_shots, dataset,
                                                                   decode_method, num_iter, num_beams, top_p)
    experiment_message = experiment_logger.generate_experiment_message(decode_method, model.tokenizer.name_or_path,
                                                                       num_shots, dataset, num_iter, num_beams, top_p)

    # 日志记录开始
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

            sorted_token_probs = model.extract_and_sort_token_probabilities(output)
            for element in sorted_token_probs[:5]:
                f.write(f"  {element[0]}: {element[1]:.4f}" + '\n')

