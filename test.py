from util import DatasetLoader
from util import Query
from util import Model
from util import ExperimentLogger


# def modify_query_info_based_on_tokens(query_info, sorted_token_probs, top_n):
#     removed_choices = [choice.replace("▁", "") for choice, _ in sorted_token_probs[:top_n]]
#     answer = query_info['final_question']['answerKey']
#     new_choices = [choice for choice in query_info['final_question']['question']['choices'] if
#                    choice['label'] not in removed_choices]
#     dict = {}
#     for i, choice in enumerate(new_choices):
#         dict[choice['label']] = chr(65 + i)
#         choice['label'] = chr(65 + i)
#     if answer in removed_choices:
#         updated_answer = 'Z'
#     else:
#         updated_answer = dict[answer]
#
#     query_info['final_question']['question']['choices'] = new_choices
#     query_info['final_question']['answerKey'] = updated_answer
#
#     return query_info
#
#
# dataset = 'commonsenseQA'
# dataset_loader = DatasetLoader()
# experiment_logger = ExperimentLogger()
#
# data = dataset_loader.load_dataset(dataset)
# query_maker = Query(data, dataset)
# query_info = query_maker.generate_shots_and_question(1)
# query_template = """Here is some example:
# {shots}
# Please solve the following question
# Question:
# {final_question}
# Option
# {final_choices}
# Incorrect answer:"""
# sorted_token_probs = [("▁A", 0)]
# modified_query_info = modify_query_info_based_on_tokens(query_info, sorted_token_probs, 1)
# modified_custom_query_info = query_maker.create_custom_query(modified_query_info, query_template,
#                                                              show_correct_answer=False)
# print(modified_custom_query_info["query"])
test = '''Initial Query:
Eliminate wrong options from the given choices for the question:
Questions:
Where does water enter in a dam to create electricity?
Options:
A: cracks
B: power turbine
C: wet clothes
D: thin soup
E: dribble
Incorrect Answers: C

Questions:
Sam was moving out to college.  His parents packed up all of his old toys are placed them in long term storage.  Where might they put the boxes of toys?
Options:
A: store
B: these twos are incorrect answers
C: bed
D: floor
E: basement
Incorrect Answers: D

Questions:
What leads a company to get in trouble?
Options:
A: procure
B: branch out
C: mail order
D: liquidated
E: commit crime
Incorrect Answers: A

Questions:
If the captain of a crew was going to an arena, what type of crew would he be in?
Options:
A: airplane cabin
B: battleship
C: military
D: basketball team
E: solider
Incorrect Answers: E

Questions:
If you needed a lamp to do your work, where would you put it?
Options:
A: desktop
B: corner
C: bedroom
D: desk
E: office
Incorrect Answers: C


Please solve the following question
Question:
Standing in queue at a store allows you to be able to do what to your surroundings?
Options:
A: watch
B: whistle
C: impatience
D: daydreaming
E: look around
Incorrect answer:'''
model = Model(model_name='meta-llama/Llama-2-7b-hf')
with open('./results_with_logits/' + "test.txt", 'w', encoding='utf-8') as f:
    for i in range(100):
        model_inputs = model.generate_input(test)
        output = model.generate_output(model_inputs, 'greedy', max_new_tokens=5)
        generated_text = model.decode_output(output)
        sorted_token_probs = model.extract_and_sort_token_probabilities(output)
        f.write(f"Iteration {i + 1}:\n")
        f.write('Initial Query:\n')
        f.write(test + '\n')
        f.write('Initial Generation: \n')
        f.write(generated_text[len(test):] + '\n')
        f.write('Initial logit: \n')
        for element in sorted_token_probs[:10]:
            f.write(f"  {element[0]}: {element[1]:.4f}" + '\n')
        f.write('Ground Truth Answer: \n E\n')
        f.write('-' * 20 + '\n\n')