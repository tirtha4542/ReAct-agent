import os
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langsmith import Client, wrappers

load_dotenv()

os.environ["LANGSMITH_API_KEY"]  = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["MISTRAL_API_KEY"]   = os.getenv("MISTRAL_API_KEY")

client = Client()

dataset_name = "Chatbot evaluation new3"
dataset = client.create_dataset(dataset_name)

client.create_examples(
    dataset_id=dataset.id,
    inputs=[
        {"question": "What is deep learning?"},
        {"question": "What is overfitting in machine learning?"},
        {"question": "What is the difference between supervised and unsupervised learning?"},
        {"question": "What is a transformer model?"},
        {"question": "What is RAG?"},
    ],
    outputs=[
        {"answer": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to learn representations of data."},
        {"answer": "Overfitting occurs when a model learns the training data too well, including noise, and performs poorly on unseen data."},
        {"answer": "Supervised learning uses labeled data to train models, while unsupervised learning finds patterns in unlabeled data."},
        {"answer": "A transformer is a deep learning architecture that uses self-attention mechanisms to process sequential data, widely used in NLP tasks."},
        {"answer": "RAG (Retrieval-Augmented Generation) combines a retrieval system with a language model to answer questions using external knowledge."},
    ]
)
print("Dataset created and examples added successfully!")

llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0,
    max_retries=2
)

# ── Chatbot function to evaluate ──────────────────────────────────────────────
def chatbot(inputs: dict) -> dict:
    """Takes dataset input, returns answer."""
    response = llm.invoke(inputs["question"])
    return {"answer": response.content}

# ── Evaluator ─────────────────────────────────────────────────────────────────
eval_instruction = "You are an expert professor specialized in grading student answers to questions."

pythoneval_instruction = "You are an expert professor specialized in grading student answers to questions."

def correctness(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict
) -> dict:
    user_content = f"""You are grading the following question:
{inputs['question']}

Here is the correct answer:
{reference_outputs['answer']}

Here is the predicted answer:
{outputs['answer']}

Respond with only 'correct' or 'incorrect':
Grade:"""

    response = llm.invoke([
        {"role": "system", "content": eval_instruction},
        {"role": "user",   "content": user_content}
    ]).content.strip().lower()

    # ✅ only key and score — no reasoning field
    return {
        "key":   "correctness",
        "score":  1 if response == "correct" else 0
    }



# ── Run evaluation ────────────────────────────────────────────────────────────
from langsmith.evaluation import evaluate

results = evaluate(
    chatbot,
    data=dataset_name,
    evaluators=[correctness],
    experiment_prefix="mistral-chatbot-v1",
    metadata={"model": "mistral-small-latest"}
)

print("Evaluation complete!")
