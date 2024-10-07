import json
import os

import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import requests
import torch
# from newspaper import Article

# creates the Dataset which I required from BGU to label
def create_unlabeled_dataset():
    from datasets import load_dataset
    from transformers import pipeline

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

# a function which saves a file of the urls of the relevant tickers
def get_urls_list():
    tickers_list = [
        "AAPL",  # Apple Inc.
        "MSFT",  # Microsoft Corporation
        "GOOGL",  # Alphabet Inc. (Class A)
        "GOOG",  # Alphabet Inc. (Class C)
        "AMZN",  # Amazon.com Inc.
        "NVDA",  # NVIDIA Corporation
        "TSLA",  # Tesla Inc.
        "META",  # Meta Platforms Inc.
        "BRK.B",  # Berkshire Hathaway Inc. (Class B)
        "UNH",  # UnitedHealth Group Incorporated
        "JNJ",  # Johnson & Johnson
        "V",  # Visa Inc.
        "XOM",  # Exxon Mobil Corporation
        "JPM",  # JPMorgan Chase & Co.
        "PG",  # Procter & Gamble Co.
        "MA",  # Mastercard Incorporated
        "HD",  # Home Depot Inc.
        "CVX",  # Chevron Corporation
        "LLY",  # Eli Lilly and Company
        "ABBV"  # AbbVie Inc.
    ]

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    results_dir = "./Articles_summaries/"
    os.makedirs(results_dir, exist_ok=True)
    results_file_name = now + ".txt"
    results_file_path = os.path.join(results_dir, results_file_name)

    urls = []
    for ticker in tickers_list:
        ticker_data = get_data(ticker, '2022-01-28', end_date='2022-02-28')
        ticker_dict = extract_relevant_articles(ticker, ticker_data)  # dict: {date: [relevant articles' urls]}
        for date, url_list in ticker_dict.items():
            for url in url_list:
                urls.append(url)
    with open(results_file_path, "w") as file:
        file.write(json.dumps(urls, indent=4))

# websites which are not compatible: Zacks
def get_text_and_summary(url):
    # Download and parse the article
    article = Article(url)
    article.download()
    article.parse()

    # Print the text of the article
    text = article.text


    # For summarization
    article.nlp()
    summary = article.summary
    return text, summary

def get_summary(url):
    article = Article(url)
    article.download()
    article.parse()

    # For summarization
    article.nlp()
    summary = article.summary
    return summary


# calculate the sentiment of the text according to the model, the text comes from get_summary(url)
def analyse_sentiment(model, tokenizer, url):
    print(f"url: {url}")
    summary = get_summary(url)
    tokenized_text = tokenizer(summary, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**tokenized_text)

    logits = outputs.logits
    predicted_class_id  = torch.argmax(logits, dim=-1).item()

    label_mapping = {0: -1, 1: 0, 2: 1} # maps the model labels to -1,0,1 labels.

    predicted_label = label_mapping[predicted_class_id]

    return predicted_label

# This function were built to help us count the days diff in order to calculate the ROI
# start_date, end_date should be a string YYYY-MM-DD
def get_index_of_date(data_dict, date):
    # Iterate through dictionary keys and count index
    for index, key in enumerate(data_dict.keys()):
        if key == date:
            return index
    # Return -1 if the date is not found
    return -1

# returns the average ROI per day: (end_price - start_price)/num_days
def get_roi_per_day(start_date, end_date, start_price, end_price, data_dict):
    num_days = get_index_of_date(data_dict,end_date) - get_index_of_date(data_dict, start_date) +1
    return (end_price - start_price)/num_days

# extracting the articles with a relevance score > 0.7
# returns a dictionary: {date: [relevant articles' urls]}
def extract_relevant_articles(ticker, data):
    # Initialize an empty dictionary
    date_url_dict = {}

    # Iterate through each article in the feed
    if 'feed' in data:
        print(data['items'])
        for article in data['feed']:
            # Extract the date and URL
            date_published = article['time_published'][:8]  # Assuming the date format is YYYYMMDD
            formatted_date = datetime.strptime(date_published, "%Y%m%d").strftime("%Y-%m-%d")
            url = article['url']

            # Check if any ticker has a relevance score higher than 0.7
            relevant_ticker_found = False
            for symbol in article.get('ticker_sentiment', []):
                if float(symbol['relevance_score']) > 0.7 and symbol['ticker'] == ticker:
                    relevant_ticker_found = True
                    break

            # If a relevant ticker is found, add the URL to the dictionary under the corresponding date
            if relevant_ticker_found:
                if formatted_date in date_url_dict:
                    date_url_dict[formatted_date].append(url)
                else:
                    date_url_dict[formatted_date] = [url]

    return date_url_dict


# gets a date in format:'2020-03-28'
def get_data(ticker, start_date, end_date):
    start_date = start_date[:4] + start_date[5:7] + start_date[8:10] + "T1200"
    end_date = end_date[:4] + end_date[5:7] + end_date[8:10] + "T1200"
    url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={}&time_from={}&time_to={}&limit=30&apikey=HQY4NMCEOKWDG9K3'.format(ticker, start_date, end_date)
    # url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={}&time_from={}&apikey=HQY4NMCEOKWDG9K3'.format(ticker, start_date)
    r = requests.get(url)
    data = r.json()
    return data


# gets a short interest file path and adds a sentiment column according to the change from previous column.
# the sentiment is for a period of 14 days.
# TODO: to check how can i create a file and save it in the env
def add_sentiment_col(file_path):
    # Load the data
    df = pd.read_csv(file_path)

    # Determine thresholds
    # Let's assume:
    #  - `a` is the 66th percentile
    #  - `b` is the 33th percentile
    a = df['% Change from Previous'].quantile(0.66)
    b = df['% Change from Previous'].quantile(0.33)


    # Map the "change from previous" to sentiment labels
    def map_to_sentiment(change):
      if change > a:
          return -1  # Negative sentiment
      elif change < b:
          return 1  # Positive sentiment
      else:
          return 0  # Neutral sentiment

    df['Sentiment'] = df['% Change from Previous'].apply(map_to_sentiment)

    return df


# gets a short_interest_path, calculate the sentiment of it, and returns the sentiment
def get_short_interest_sentiment(short_interest_path, settel_date):

    short_interest_file = add_sentiment_col(short_interest_path)

    # Convert the 'Settlement Date' column to datetime format
    short_interest_file['Settlement Date'] = pd.to_datetime(short_interest_file['Settlement Date'], format='%m/%d/%Y')

    # Filter the DataFrame to get the row corresponding to the date '8/31/2023'
    date_of_interest = pd.Timestamp(settel_date)
    filtered_row = short_interest_file[short_interest_file['Settlement Date'] == date_of_interest]

    # Extract the '% Change from Previous' value for the specified date
    sentiment = filtered_row['Sentiment'].values[0] if not filtered_row.empty else None

    return sentiment


# Evaluation VS short-interest for 14 days
# gets a date in format:'2024-03-28' / '3/28/2024'
# the short_interest_path should lead to the "equityshortinterest_AAPL_NNM_with_sentiment" which is the short_interest file after adding the sentiment, using the add_sentiment_col(file_path)
def model_vs_short_interest(ticker, model,tokenizer, short_interest_file, settel_date):

    # Convert the input settlement date to datetime
    settel_date = pd.to_datetime(settel_date)

    # Ensure the 'Settlement Date' column is in datetime format and strip any leading/trailing whitespace
    short_interest_file['Settlement Date'] = pd.to_datetime(short_interest_file['Settlement Date'].str.strip())

    short_interest_file = short_interest_file.sort_values(by='Settlement Date').reset_index(drop=True)

    # Find the index of the provided settlement date
    settel_date_idx = short_interest_file.index[short_interest_file['Settlement Date'] == settel_date].tolist()

    if not settel_date_idx:
        return "Settlement date not found in the file."

    settel_date_idx = settel_date_idx[0]

    if settel_date_idx == 0:
        return "No previous settlement date record available."

    # Get the previous settlement date record
    start_date = short_interest_file.iloc[settel_date_idx - 1]['Settlement Date']

    # Convert Timestamps to strings
    start_date_str = start_date.strftime('%Y-%m-%d')
    settel_date_str = settel_date.strftime('%Y-%m-%d')

    data = get_data(ticker, start_date_str, settel_date_str)
    date_url_dict = extract_relevant_articles(ticker, data)

    scores = []
    for date, urls in date_url_dict.items():
      day_scores = [analyse_sentiment(model, tokenizer, url) for url in urls]
      average = sum(day_scores) / len(day_scores) if day_scores else 0
      scores.append(average)


    sum_scores = 0
    for score in scores:
        sum_scores += score
    total_score = sum_scores / len(scores)
    if total_score >= 0.5:
      model_analysis =  1
    elif total_score >= -0.5:
      model_analysis =  0
    else: model_analysis =  -1

    # CHECK MODEL_ANALYSIS VS SHORT_INTEREST_SENTIMENT:
    short_interest_sentiment = get_short_interest_sentiment(short_interest_file, settel_date)

    print(f"The model sentiment is: {model_analysis}.")
    print(f"The Short-Interest-File sentiment is: {short_interest_sentiment}.")

# short_interest_with_sentiment = add_sentiment_col('/Data/equityshortinterest_AAPL_NNM.csv')
# model_name = "distilbert-base-uncased-finetuned-sst-2-english"
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model_vs_short_interest('AAPL', model, tokenizer, short_interest_with_sentiment, '2024-03-28')


from transformers import pipeline

# Load the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Load your article text (example)
# If you have a file, read the content as follows:
# with open("article.txt", "r", encoding="utf-8") as file:
#     article_text = file.read()



# # For demonstration, let's use a sample text directly
# article_text = """
# SAN DIEGO (AP) _ Retrophin Inc. (RTRX) on Tuesday reported a loss of $17.6 million in its fourth quarter. On a per-share basis, the San Diego-based company said it had a loss of 55 cents. Earnings, adjusted for non-recurring costs and pretax expenses, came to 7 cents per share. The drug developer posted revenue of $42.2 million in the period. For the year, the company reported that its loss widened to $59.7 million, or $1.54 per share. Revenue was reported as $154.9 million. Retrophin shares have risen 13 percent since the beginning of the year. In the final minutes of trading on Tuesday, shares hit $23.88, a rise of 13 percent in the last 12 months. This story was generated by Automated Insights ( http://automatedinsights.com/ap ) using data from Zacks Investment Research. Access a Zacks stock report on RTRX at https://www.zacks.com/ap/RTRX
# """
#
# # Summarize the article
# summary = summarizer(article_text, max_length=100, min_length=80, do_sample=False)
#
# # Display the summary
# print("Summary:")
# print(summary[0]['summary_text'])

print("yo")