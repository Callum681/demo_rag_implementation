## Example LM fine tune

# fine_tune_tinyllm.py
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

# Load model and tokenizer
model_name = "distilgpt2"  # small & fast for demo
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Example dataset: short Q&A pairs
data = {
    "train": [
        {"prompt": "What is AI?", "answer": "Artificial intelligence is the simulation of human intelligence in machines."},
        {"prompt": "Define machine learning.", "answer": "Machine learning is the field that gives computers the ability to learn from data."},
    ],
    "test": [
        {"prompt": "What is deep learning?", "answer": "A subset of ML that uses neural networks with many layers."}
    ]
}

# Convert to Hugging Face Dataset
from datasets import Dataset
train_dataset = Dataset.from_list(data["train"])

# Tokenize function
def tokenize(batch):
    inputs = [f"Q: {p}\nA: {a}" for p, a in zip(batch["prompt"], batch["answer"])]
    tokenized = tokenizer(inputs, truncation=True, padding="max_length", max_length=64)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

train_dataset = train_dataset.map(tokenize, batched=True)

# Training config
args = TrainingArguments(
    output_dir="./tinyllm_finetuned",
    per_device_train_batch_size=2,
    num_train_epochs=2,
    logging_steps=5,
    save_strategy="no"
)

trainer = Trainer(model=model, args=args, train_dataset=train_dataset)
trainer.train()

# Save the model
model.save_pretrained("./tinyllm_finetuned")
tokenizer.save_pretrained("./tinyllm_finetuned")
print("✅ Fine-tuning complete and model saved to ./tinyllm_finetuned")
