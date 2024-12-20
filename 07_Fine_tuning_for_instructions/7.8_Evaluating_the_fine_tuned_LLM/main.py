from data_setup.data_preprocessing import test_data

import urllib.request
from tqdm import tqdm
import json


# First step:
# "ollama serve" in terminal
# The model which we will serve is the "llama3" model. 
# llama3 is an 8B parameter model that is instruction fine tuned.
#   Model
#     architecture        llama    
#     parameters          8.0B     
#     context length      8192     
#     embedding length    4096     
#     quantization        Q4_0     

file_path = "instruction-data-with-response.json"

with open(file_path, "r") as file:
    test_data = json.load(file)


def query_model(
    prompt,
    model="llama3",
    url="http://localhost:11434/api/chat"
):
    # Create the data payload as a dictionary
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {     # Settings below are required for deterministic responses
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }

    # Convert the dictionary to a JSON formatted string and encode it to bytes
    payload = json.dumps(data).encode("utf-8")

    # Create a request object, setting the method to POST and adding necessary headers
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST"
    )
    request.add_header("Content-Type", "application/json")

    # Send the request and capture the response
    response_data = ""
    with urllib.request.urlopen(request) as response:
        # Read and decode the response
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]

    return response_data


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


def generate_model_scores(json_data, json_key, model="llama3"):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score = query_model(prompt, model)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score: {score}")
            continue

    return scores



# model = "llama3"
# result = query_model("What do Llamas eat?", model)
# print(result)


# for entry in test_data[:3]:
#     prompt = (
#         f"Given the input `{format_input(entry)}` "
#         f"and correct output `{entry['output']}`, "
#         f"score the model response `{entry['model_response']}`"
#         f" on a scale from 0 to 100, where 100 is the best score. "
#     )
#     print("\nDataset response:")
#     print(">>", entry['output'])
#     print("\nModel response:")
#     print(">>", entry["model_response"])
#     print("\nScore:")
#     print(">>", query_model(prompt))
#     print("\n-------------------------")


# The Llama 3 model provides a reasonable evaluation and also gives partial points if a model is not entirely correct,
# as we can see based on the "cumulus cloud" answer. Note that the previous prompt returns very verbose evaluations; 
# we can tweak the prompt to generate integer responses in the range between 0 and 100 (where 100 is best) to calculate an average score for our model


scores = generate_model_scores(test_data, "model_response")
print(f"Number of scores: {len(scores)} of {len(test_data)}")
print(f"Average score: {sum(scores)/len(scores):.2f}\n")


# TODO:
# Exercise 7.4 Parameter-efficient fine-tuning with LoRA
# To instruction fine-tune an LLM more efficiently, modify the code in this chapter to
# use the low-rank adaptation method (LoRA) from appendix E. Compare the training
# run time and model performance before and after the modification.