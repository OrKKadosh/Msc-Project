import os
from datasets import load_dataset
from transformers import pipeline

# creates the Dataset which I required from BGU to label
def create_unlabeled_dataset():


    # Function to filter and get the dataset
    def get_filtered_articles():
        # Load the dataset from Hugging Face
        dataset = load_dataset("Lettria/financial-articles", split="train")

        # Define a function to filter articles based on conditions
        def filter_articles(example):
            # Check if the 'origin' is 'www.cnbc.com' and content length is between 140 and 200 words
            content_length = len(example['content'].split())
            return (example['origin'] == 'www.cnbc.com') and (130 <= content_length <= 250)

        # Apply the filter to the dataset
        filtered_dataset = dataset.filter(filter_articles)

        # Select only 1000 samples
        filtered_dataset = filtered_dataset.shuffle(seed=42).select(range(min(1000, len(filtered_dataset))))

        return filtered_dataset

    # Get the filtered dataset
    filtered_articles_dataset = get_filtered_articles()

    # Initialize the summarization pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Function to generate a summary for each article
    def generate_summary(example):
        try:
            summary_text = summarizer(example['content'], max_length=100, min_length=80, do_sample=False)
            return {"summary": summary_text[0]['summary_text']}
        except Exception as e:
            # Return an error message or empty string in case of exception
            return {"summary": "error"}

    # Add the summary column to the dataset
    filtered_articles_dataset = filtered_articles_dataset.map(generate_summary)
    df = filtered_articles_dataset.to_pandas()

    results_dir = "./Articles_summaries/"
    os.makedirs(results_dir, exist_ok=True)
    results_file_name = "Dataset.csv"
    results_file_path = os.path.join(results_dir, results_file_name)
    # Save the DataFrame to a CSV file
    df.to_csv(results_file_path, index=False)

    print(f"Dataset saved successfully at {results_file_path}")