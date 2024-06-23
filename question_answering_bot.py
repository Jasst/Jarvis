import asyncio
import requests
from bs4 import BeautifulSoup
from transformers import pipeline, DistilBertForQuestionAnswering, DistilBertTokenizerFast
import json
import os

# Load the fine-tuned model and tokenizer
model_name = "./finetuned_model"
model = DistilBertForQuestionAnswering.from_pretrained(model_name)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
nlp = pipeline("question-answering", model=model, tokenizer=tokenizer)


# Function to perform web search
async def search_with_google(prompt):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(f"https://www.google.com/search?q={prompt}", headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extracting all the URLs from the search results
    links = []
    for item in soup.find_all('a'):
        href = item.get('href')
        if href and 'url?q=' in href:
            link = href.split('url?q=')[1].split('&')[0]
            links.append(link)

    return links[:5]  # Return top 5 links


# Function to extract text from a web page
async def extract_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        page_text = ' '.join([para.get_text() for para in paragraphs])
        return page_text
    except:
        return ""


# Function to analyze text with transformer model
async def analyze_with_transformer(prompt, context):
    result = nlp(question=prompt, context=context)
    return result['answer']


# Function to store data
def store_data(prompt, answer):
    data = {}
    if os.path.exists("data.json"):
        with open("data.json", "r", encoding="utf-8") as file:
            data = json.load(file)

    data[prompt] = answer

    with open("data.json", "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


# Main function
async def main(prompt):
    # Perform web search
    links = await search_with_google(prompt)

    # Extract text from top links
    contexts = await asyncio.gather(*[extract_text_from_url(link) for link in links])
    contexts = [context for context in contexts if context]  # Filter out empty results

    # Analyze text with transformer model
    answers = await asyncio.gather(*[analyze_with_transformer(prompt, context) for context in contexts])

    # Combine and store answers
    combined_answer = ' '.join(answers)
    store_data(prompt, combined_answer)

    print(f"Combined Answer: {combined_answer}")


# Get user input and run main function
prompt = input("Введите запрос: ")
asyncio.run(main(prompt))
