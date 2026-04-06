---
Title: From Prototype to Pipeline: Building a Financial Sentiment Analyzer with FinBERT and RAG (by Group "Finword Warriors")
Date: 2026-03-28 17:12
Category: Reflective Report
Tags: Group Finword Warriors
---

By Group Finword Warriors

## Introduction

Emotions are natural instincts for every human being, which becomes crucial in the stock market too. Negative news will create panic feelings about a company's future, such as the ongoing Middle East crisis involving the U.S., Israel, and Iran, which pushes the selling pressure and stock price drops, resulting in market risk aversion. Observing more uncertainties and news overflowing on the Internet, it is necessary to develop a professional sentiment analyzer to interpret news trends and market feelings, for better tradings on trends.

In this blog, we would like to share about our sentiment analyzer – Sentiment Sage, including why we would like to make this app and the basic working principles and improvement we would do throughout the project.

## Motivation

**Sentiment is the new 'trillion dollar question'**

Living in a volatile and continuously-changing society, basic analysis of a company is not always the optimal strategy nowadays. Looking at Hang Seng Tech dropping over 10% in a month and the global market situation, like reciprocal tariffs last year, it is assured that a single quote by political leaders carrying different emotions can drive the market trends up and down like a roller coaster. Sentiments will be the major pillar of short-term trades and options. Observed from recent chaotic situations like wars and a Citrini post criticizing the AI bubble and intelligent crisis, pushing down the Dow Jones index of over 2%, we can further confirm that sentiment is the new fundamental analysis, it matters like a 'trillion dollar question'. Integrating with so much information by different news companies overflowing around the world, a normal Gemini or Grok would not be able to correctly interpret the current situation, that's why we develop this sentiment analyzer that aims to provide users with reliable and data-driven sentiment insights.

More AI apps are published these days like Bevel and Backbone AI to analyze market emotions, but they still have certain advantages. For example, their news information is not updated in real time compared to our up-to-date news thanks to the newsAPI key, which matters a lot for an information overload world nowadays. Besides, those AI are not specific to interpret financial terms like 'bear' and 'default', which might lead to huge mistakes.

That's why we believe our model would stand out. Targeting the pain points of generic AI and apps, SentimentSage addresses the reliability gap in retail financial tech, ensuring that AI-generated insights are both timely and grounded in reality.

## Approaches

The development of SentimentSage will follow a structured approach focused on accuracy, usability, and scalability. The project will begin with a prototype using FinBERT, a pre-trained NLP model specifically designed for financial news sentiment analysis. FinBERT is chosen over other general models because of its deep understanding of financial terminology and context, resulting in more reliable sentiment classification.

```python

def get_finbert_pipeline():
    model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
	return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

```

To enhance the quality of results, Retrieval-Augmented Generation (RAG) is integrated into the system, combining information retrieval with generative LLM AI. To initiate this process, the system automatically translates the user-provided company ticker into optimized search queries for the News API. This ensures that the RAG pipeline operates on the most recent and contextually relevant financial headlines, bridging the gap between raw market data and actionable intelligence.

These retrieved articles are then processed and passed as context to the LLM, ensuring that the generated summaries are strictly grounded in real-time data. By synthesizing these diverse information sources, the system can provide a comprehensive market analysis that goes far beyond the capabilities of a standalone language model.

```python

loader = TextLoader("temp_articles.txt", encoding="utf-8")

documents = loader.load()

splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

docs = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(docs, embeddings)

```

The system accepts user input in the form of a company ticker, retrieves recent news articles, analyzes sentiment using FinBERT, and generates a comprehensive summary with actionable insights.

For deployment, the application will be migrated to Streamlit Cloud, providing a clean user interface. API keys for LLM and News API must be obtained and managed using Streamlit's secrets management section.

The project achieves a sophisticated synergy by combining FinBERT's quantitative sentiment scores with the LLM's qualitative reasoning. By treating the sentiment probabilities as structured features for the LLM, SentimentSage transitions from simple emotion detection to a "rational" synthesis, providing users with a comprehensive summary grounded in both market mood and factual evidence.

## Challenges and Limitation

**Single LLM API Key?**

```python

from langchain_openai import OpenAI, OpenAIEmbeddings

import os

os.environ["OPENAI_API_KEY"] = "your-openai-key"

llm = OpenAI(temperature=0)    # 硬编码 OpenAI

embeddings = OpenAIEmbeddings()

```

We first used OpenAI's API for convenience when we initially constructed our RAG pipeline. It worked, but it can be inconvenient for those who cannot use or are not willing to use OpenAI's API. Moreover, it could be inflexible to tie our project to a single API supplier. Therefore, we realized we should modify the LLM interface if we wanted to use a self-hosted model or a free substitute like Kimi. Instead of using langchain_openai directly, the model name, API key, and base URL should be read from environment variables in the LLM interface. In other words, any base URL and API key provider that are compatible with LLM interface can be used.

## Evolutions

**Replacing and Integrating More LLM API Keys**

Here's the simplified setup we added:

```python

import os

from langchain_openai import ChatOpenAI

def get_llm():

    return ChatOpenAI(

    model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),

    openai_api_key=os.getenv("OPENAIA_PI_KEY"),

    openai_api_base=os.getenv("OPENAIA_API_BASE", "https://api.openai.com/v1"),

    temperature=0

    )

# Kimi as an example

os.environ["OPENAI_API_KEY"]="your_api_key"

os.environ["OPENAI_API_BASE"] = "https://api.moonshot.cn/v1"

os.environ["LLM_MODEL"] = "moonshot-v1-8k"

# Testing

llm = get_llm()

response = llm.invoke("Introduce yourself")

print(response.content)  # Sucessful

```

We only need to set three environment variables in order to use Kimi or another service. No modifications to the code.

The biggest obstacle was to ensure the response format complied with LangChain's requirements. Slightly different JSON structures returned by some non-OpenAI endpoints may raise an error. So we have another choice, which is the ChatOpenAI class, adaptable enough to work with the majority of OpenAI-compatible APIs. Our project became future-proof thanks to this minor refactor, which also allowed us to experiment with different LLMs without having to rewrite any code. It serves as a lesson in creating modular, adaptable systems from the outset.

## Reference

Citrini (22 Feb 2026) The 2028 Global Intelligence Crisis https://www.citriniresearch.com/p/2028gic

