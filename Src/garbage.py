import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the tokenizer and model
model_name = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Automatically maps parts of the model to available devices
    torch_dtype="auto"  # Uses the most appropriate dtype (e.g., float16 on GPUs)
)


def generate_synthetic_sample(prompt, max_length=200, num_return_sequences=1, temperature=0.7):
    # Tokenize the input with padding and truncation
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length  # Explicitly enforce max length during tokenization
    ).to(model.device)
    # Generate outputs with the pad token explicitly set
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,  # Provide the attention mask
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        temperature=temperature,  # Controls creativity
        top_k=50,  # Limits the sampling pool for more coherent text
        top_p=0.9,  # Nucleus sampling
        do_sample=True,  # Enables sampling
        pad_token_id=tokenizer.pad_token_id,  # Ensure the model handles padding properly
    )
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


# Example prompt
prompt = """Please generate a financial text with a neutral sentiment, here are some examples
1. According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing .
2. In Sweden , Gallerix accumulated SEK denominated sales were down 1 % and EUR denominated sales were up 11 % .
3. When this investment is in place , Atria plans to expand into the Moscow market .
4. In June it sold a 30 percent stake to Nordstjernan , and the investment group has now taken up the option to acquire EQT 's remaining shares .
5. The new plant is planned to have an electricity generation capacity of up to 350 megawatts ( MW ) and the same heat generation capacity .
"""
synthetic_samples = generate_synthetic_sample(prompt, max_length=200, num_return_sequences=3)

# Print generated samples
for idx, sample in enumerate(synthetic_samples, 1):
    print(f"Sample {idx}: {sample}")
