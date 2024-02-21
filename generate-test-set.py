import dotenv
import os, re
from openai import AzureOpenAI
import pandas as pd
from tqdm import tqdm  # Import tqdm

dotenv.load_dotenv()

openai_api_type = os.getenv("AZURE_OPENAI_API_TYPE")
openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
openai_gpt4_api_base = os.getenv("AZURE_OPENAI_GPT4_API_BASE")
openai_gpt4_api_key = os.getenv("AZURE_OPENAI_GPT4_API_KEY")
openai_gpt4_deployment_name = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT_NAME")

generator_client = AzureOpenAI(
    azure_endpoint=openai_gpt4_api_base,
    api_key=openai_gpt4_api_key,
    api_version=openai_api_version
)

# Load ground truth from CSV
df = pd.read_csv("dummy.csv")

# System prompt as per your requirement
system_prompt = """You are an assistant tasked with creating a dataset for testing purposes. Please create a set of questions based on the provided ground truth below. Make up to 5 questions based on the ground truth, ensuring that the questions you create can be answered with the information from the ground truth. All questions should be formulated in Indonesian."""

responses = []

try:
    # Directly wrap the Series with tqdm for a progress bar
    for index, ground_truth in tqdm(enumerate(df['Revised Content']), total=df.shape[0], desc="Processing"):
        try:
            user_prompt = f"""ground truth: {ground_truth}"""
            
            completion = generator_client.chat.completions.create(
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_prompt}],
                model=openai_gpt4_deployment_name,
                temperature=0,
                max_tokens=1000,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                seed=42
            )

            response = completion.choices[0].message.content.strip()
            questions = re.sub(r'^\d+\.\s*', '', response, flags=re.MULTILINE).split('\n')
            # questions = response.split('\n')
            for question in questions:
                responses.append([ground_truth, question.strip()])
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            continue
except KeyboardInterrupt:
    print("Interrupted by user, saving progress...")

# Save responses to CSV
output_df = pd.DataFrame(responses, columns=['ground_truth', 'question'])
output_df.to_csv('data-set.csv', index=False)
print("Progress saved to data-set.csv")

# ground_truth = "BRI didirikan di Jakarta di Indonesia"

# system_prompt = """You are an assistant tasked with creating a dataset for testing purposes. Please create a set of questions based on the provided ground truth below. Make up to 5 questions based on the ground truth, ensuring that the questions you create can be answered with the information from the ground truth."""

# user_prompt = f"""ground truth: {ground_truth}"""

# completion = generator_client.chat.completions.create(
#     messages = [{"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt}],
#     model = openai_gpt4_deployment_name,
#     temperature = 0,
#     max_tokens = 1000,
#     top_p = 1,
#     frequency_penalty = 0,
#     presence_penalty = 0,
#     stop = None,
#     seed = 42
# )

# print(system_prompt, "\n")

# response = completion.choices[0].message.content

# print(response)
