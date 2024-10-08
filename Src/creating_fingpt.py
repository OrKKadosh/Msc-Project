from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizerFast
from peft import PeftModel  # 0.5.0
import re

# Function to clean the output from unknown symbols
def clean_output(text):
    # Remove non-printable or non-ASCII characters
    return re.sub(r'[^\x20-\x7E]+', '', text)

# Load Models
base_model = "NousResearch/Llama-2-13b-hf"
peft_model = "FinGPT/fingpt-sentiment_llama2-13b_lora"
tokenizer = LlamaTokenizerFast.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = LlamaForCausalLM.from_pretrained(base_model, trust_remote_code=True, device_map="cuda:0", load_in_8bit=True)
model = PeftModel.from_pretrained(model, peft_model)
model = model.eval()

# Make prompts
prompt = [
    '''Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}
    Input: The Board of Directors of CDK Global, Inc. has declared a regular quarterly cash dividend of $0.15 per share. The dividend is payable on March 29, 2018 to shareholders of record at the close of business on March 1, 2018. With more than $2 billion in revenues, CDK is a leading global provider of integrated information technology and digital marketing solutions to the automotive retail and adjacent industries.
    Answer: ''',
    '''Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}
    Input: Ashley J. Jericho has over nine years of insolvency experience. She has represented consumer and business debtors, creditors and trustees in bankruptcy proceedings. Jericho can be reached at ajericho@mcdonaldhopkins.com or 248.593.2945 X1945.About McDonald Hopkins: McDonald Hopkins is a business advisory and advocacy law firm with locations in Chicago, Cleveland, Columbus, Detroit, Miami and West Palm Beach.
    Answer: ''',
    '''Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}
    Input: Tanzania's energy regulator raises maximum price for petrol, diesel and kerosene. Fuel prices have a big influence on the inflation rate in the East African country. In the 12 months through November inflation eased to 4.4 percent, from 5.1 percent the previous month. Analysts say the fuel price hikes could push the inflation rates up again, analysts said in a report.
    Answer: ''',
]

# Tokenize with truncation
tokens = tokenizer(prompt, return_tensors='pt', padding='max_length', max_length=512, truncation=True).to('cuda')

# Generate results, using max_new_tokens instead of max_length
res = model.generate(**tokens, max_new_tokens=50, pad_token_id=tokenizer.pad_token_id)

# Decode the output
res_sentences = [tokenizer.decode(i, skip_special_tokens=True) for i in res]
out_text = [o.split("Answer: ")[1] for o in res_sentences if "Answer: " in o]

# Clean each result to remove the unknown symbols
cleaned_out_text = [clean_output(sentiment) for sentiment in out_text]

# Show results
for sentiment in cleaned_out_text:
    print(sentiment)

# positive
# neutral
# negative

# CODE FOR RUNNING THE MODEL
for model in base_models:
    tokenizer = LlamaTokenizerFast.from_pretrained(model["tokenizer"], trust_remote_code=True)

    if model["name"] == "FinGPT-fingpt-sentiment-llama2-13b":
        preprocess_function = lambda examples: preprocess_instruction_based_dataset(examples, tokenizer)
        compute_metrics_function = compute_metrics_for_instruction_based
    else:
        preprocess_function = lambda examples: tokenize_function(tokenizer, examples)
        compute_metrics_function = compute_metrics

    # Process training and test datasets with the respective preprocess function
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

    trainer = Trainer(
        model=model["model"],
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_function
    )

    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Model: {model['name']}, Accuracy: {eval_results['eval_accuracy']}")
