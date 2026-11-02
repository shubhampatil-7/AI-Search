from search_utils import RRF_K, load_movies
from hybrid_search import rrf_search_command

import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
model = "openrouter/free"

def augment(query, limit) -> None:
    # movies = load_movies()
    res = rrf_search_command(query, RRF_K, None, limit)
    docs = res["results"]

    prompt = f"""You are a RAG agent for Webflyx, a movie streaming service.
        Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
        Provide a comprehensive answer that addresses the user's query.

        Query: {query}

        Documents:
        {docs}

        Answer:"""

    response = client.chat.completions.create( model=model, messages=[{"role": "user", "content": prompt}]
    )

    llm_res = (response.choices[0].message.content or "").strip().strip('"')

    print("Search Results:\n")
    for i in res["results"]:
        print(f"- {i['title']}")
    print("\n")
    print("RAG Response:\n")
    print(llm_res)

def summarize(query, limit) -> None:
    # movies = load_movies()
    res = rrf_search_command(query, RRF_K, None, limit)
    results = res["results"]

    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

    The goal is to provide comprehensive information so that users know what their options are.
    Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

    This should be tailored to Webflyx users. Webflyx is a movie streaming service.

    Query: {query}

    Search results:
    {results}

    Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

    response = client.chat.completions.create( model=model, messages=[{"role": "user", "content": prompt}]
    )

    llm_res = (response.choices[0].message.content or "").strip().strip('"')

    print("Search Results:\n")
    for i in results:
        print(f"    - {i['title']}")
    print("\n")
    print("LLM Summary:\n")
    print(llm_res)


def cite(query, limit)-> None:
    docs = rrf_search_command(query, RRF_K, None, limit)
    documents = docs["results"]

    prompt = f"""Answer the query below and give information based on the provided documents.

    The answer should be tailored to users of Webflyx, a movie streaming service.
    If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

    Query: {query}

    Documents:
    {documents}

    Instructions:
    - Provide a comprehensive answer that addresses the query
    - Cite sources in the format [1], [2], etc. when referencing information
    - If sources disagree, mention the different viewpoints
    - If the answer isn't in the provided documents, say "I don't have enough information"
    - Be direct and informative

    Answer:"""

    response = client.chat.completions.create( model=model, messages=[{"role": "user", "content": prompt}]
        )

    llm_res = (response.choices[0].message.content or "").strip().strip('"')
    
    print("Search Results:\n")
    for i in documents:
        print(f"- {i['title']}")
    print("\n")
    print("LLM Answer:\n")
    print(llm_res)