import time
import transformers
import torch
import pandas as pd
import json
from transformers import AutoTokenizer
from transformers import pipeline

model = "meta-llama/Llama-2-7b-chat-hf" 
tokenizer = AutoTokenizer.from_pretrained(model, token= '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' )

llama_pipeline = pipeline(
    "text-generation",  # LLM task
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    token = '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'
)

def load_data(file_path):
    """
    Load data from a JSONL file and create a DataFrame.

    Parameters:
        file_path (str): The path to the JSONL file.

    Returns:
        DataFrame: A DataFrame containing the loaded data.
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    return df

def create_prompts(num_examples, question, choices, df):
    """
    Create a prompt for a given question with multiple choices and correct answers.

    Parameters:
        num_examples (int): The number of examples to include.
        question (str): The question text.
        choices (dict): A dictionary containing the choices (options) for the question.
        df (DataFrame): The DataFrame containing the data.

    Returns:
        str: A formatted prompt string.
    """
    prompt = "" 
    if num_examples != 0: 
        df_ex = df[-num_examples:]
        # prompt += "\033[1mInstruction\033[0m: Like the examples below, fill in the answer blank with the correct choice:\n"
        # prompt += "Like the " + str(num_examples) + " examples below, answer the last question with a brief explanation"
        # prompt += 'Choose an answer for the last question with a reason.\n'
        for i in range(1, len(df_ex)+1):
            stem = df_ex['question'][len(df)-i]['stem']
            options = {choice['label']: choice['text'] for choice in df['question'][len(df)-i]['choices']}
            correct_ans = df['answerKey'][len(df)-i]
            prompt+= "Question: " + stem + "\nOptions:\n" + \
                     "A " + options['A'] + "\n" + \
                     "B " + options['B'] + "\n" + \
                     "C " + options['C'] + "\n" + \
                     "D " + options["D"] + "\n" + \
                     "E " + options["E"] + "\n" + \
                     "Correct Answer " + correct_ans + "\n" 
        prompt += "\033[1mFill in the correct option for the below question from the given choices like the above examples with a brief explanation\033[0m\n"
    prompt += "Question: " + question + "\nOptions:\n" + \
             "A " + choices['A'] + "\n" + \
             "B " + choices['B'] + "\n" + \
             "C " + choices['C'] + "\n" + \
             "D " + choices["D"] + "\n" + \
             "E " + choices["E"] + "\n" + \
             "Correct Answer: "
    return prompt

def get_llama_response(prompt: str, decoding_method = 'sampling', top_p=0.97, top_k=0, num_beams=3, temperature = 1, max_length=300) -> None:
    """
    Generate a response from the Llama model.

    Parameters:
        prompt (str): The user's input/question for the model.
        decoding_method (str): The decoding method to use. Options are 'sampling', 'greedy', or 'beam'. Default is 'sampling'.
        top_p (float): Top-p sampling parameter. Default is 0.97.
        top_k (int): Top-k sampling parameter. Default is 0.
        num_beams (int): Number of beams for beam search decoding. Default is 3.
        temperature (float): Temperature parameter for sampling. Default is 1.
        max_length (int): Maximum length of the generated sequences. Default is 300.

    Returns:
        str: The model's response.
    """
    if decoding_method == "sampling":
        sequences = llama_pipeline(
            prompt,
            do_sample=True,
            top_p= top_p,
            top_k= top_k,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=max_length,
            temperature=temperature,
        )
    elif decoding_method == "greedy":
        sequences = llama_pipeline(
            prompt,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            max_length=max_length,
            temperature=temperature,
        )
    elif decoding_method == "beam":
        sequences = llama_pipeline(
            prompt,
            num_beams=num_beams,
            eos_token_id=tokenizer.eos_token_id,
            max_length=max_length,
            temperature=temperature,
        )
    else:
        raise ValueError("Invalid decoding method. Choose from 'sampling', 'greedy', or 'beam'.")

    res = sequences[0]['generated_text']
    return res


def extract_ans(ans):
    """
    Extract the answer from the generated response.

    Parameters:
        ans (str): The generated response.

    Returns:
        str: The extracted answer.
    """
    lines = ans.split()
    for i in range(len(lines)):
        if lines[i] == 'Answer:' and i + 1 < len(lines):  # Check if the next line exists
            res = lines[i + 1]
            res = res.replace('.', '')
            return res
    return None

def get_accuracy(df, num_examples, decoding_method, iterations):
    """
    Calculate the accuracy of the Llama model.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
        num_examples (int): The number of examples to include.
        decoding_method (str): The decoding method to use.
        iterations (int): The number of iterations for calculating accuracy.

    Returns:
        None: Prints the accuracy and elapsed time.
    """
    num_correct = 0
 
    start_time = time.time()
    for i in range(iterations):
        stem = df['question'][i]['stem']
        choices = {choice['label']: choice['text'] for choice in df['question'][i]['choices']}
        correct_ans = df['answerKey'][i]
        prompt = create_prompts(num_examples,stem, choices,df)
        if num_examples == 0:
            ans = get_llama_response(prompt, decoding_method = decoding_method, top_p=0.97, top_k=5, temperature = 1, max_length=100)
        if num_examples == 2:
            ans = get_llama_response(prompt, decoding_method = decoding_method, top_p=0.97, top_k=5, temperature = 1, max_length=200)
        if num_examples == 5:
            ans = get_llama_response(prompt, decoding_method = decoding_method, top_p=0.97, top_k=5, temperature = 1, max_length=400)
        print(ans)
        print("---------------------------------------------------------------------------")
        gen_ans = extract_ans(ans)
        print(gen_ans)
        if gen_ans == correct_ans:
            num_correct+=1
         

    end_time = time.time()
    print("The accuracy of {} shot is: {}".format(num_examples, num_correct/iterations))
    elapsed_time = end_time - start_time
    print("Time taken for {} shot prompting using decoding style {} is {} seconds:".format(num_examples, decoding_method, elapsed_time))

df = load_data('train_rand_split.jsonl')
for num_examples in [0,2,5]:
    get_accuracy(df, num_examples, 'sampling', 300)