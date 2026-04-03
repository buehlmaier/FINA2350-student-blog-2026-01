---
Title: Building a Pre-IPO Hype Index: Social Sentiment as a Predictor of IPO Underpricing - Blog #1 (by Group Gen the Alpha)
Date: 2026-03-27 12:00
Category: Reflective Report
Tags: Group Gen the Alpha
---

By Group "Gen the Alpha"

# Background

In our modern financial markets, most financial analysts and investors rely on historic stock prices and patterns when devising their investment strategies on existing stocks. However, there are no historical stock data for new IPOs, making predictions and trend analysis near impossible. We aim to address this by providing sentiment analysis of upcoming IPOs to help predict immediate returns at listing by analysing online and public discussions during the preceding period. 

![Sentiment on Stock]({static}/images/Gen-the-Alpha-01-The-Role-of-Market-Sentiment-in-Stock-Price-Movements.jpg)

After an IPO is announced, investors will start online discussions about a company’s potential, financial news will cover the company's movements, and pre-IPO documents, such as SEC filings, will be available online. By leveraging NLP (natural language processing) to extract and analyse these valuable sources and textual data, we can dissect the sentiment conveyed by those texts and then use statistical methods to make predictions.

For our project, we will perform backtesting on IPOs from 2023 to 2024 to determine whether we can use hype on social media and in financial news to predict IPOs' first-day return trends. In the following blog posts, data sources for social media posts, news, and financial data, as well as selection criteria for IPOs and our methodologies, will be discussed. 

In this blog post, we will detail our selected social media data sources and journey in developing an accessible data collection method. 

# Group Introduction

Group Members and Team Introduction

**Brian (Actuarial Science)**: Focuses on defining the study's core variables from a market-structure perspective, deciding which IPOs to include, how to define the event windows (announcement to listing), and leading the preprocessing and cleaning of social media posts. Brian is also involved in designing and running the financial back-testing to see how well our sentiment signals would have performed historically.

**Brandon (Fintech)**: Concentrates on the financial side of variable selection. Determines what market and accounting data should be collected (e.g., prices, volumes, returns, and other fundamentals) and takes charge of cleaning and organising news articles. Also contributes to the financial back-testing framework, comparing sentiment-driven strategies against benchmark performance.

**Hugo (IBGM)**: Responsible for the social media data workstream, including identifying and setting up reliable data sources for social media posts (such as specific platforms and APIs), then collecting and validating that data. Also works on transforming raw posts into usable features and contributes to building the initial sentiment and prediction models.

**Stephen (Fintech)**: Focuses on the news and media data pipeline. Identifies suitable news sources and APIs, collects and validates those datasets, and engineers features that capture news tone and timing. Also involved in integrating these features into the modelling process and evaluating how news sentiment interacts with social media sentiment. Additionally, supports the collection and development of social media data and sentiment analysis features. 

**Kelvin (Fintech)**: In charge of the financial market data stream. He determines where and how to source reliable financial data (prices, IPO information, and trading metrics), performs quality checks, and constructs the variables needed for empirical analysis. Also works closely on feature construction and model building, ensuring that sentiment measures can be meaningfully linked to actual IPO performance. Additionally, supports the collection and development of news and media data, as well as sentiment analysis features.

# Data Sources

To determine how market sentiment affects IPO underpricing, we are pulling in three different types of data: social media and news data to measure public sentiment, and financial data to actually backtest our findings.

# Social Media Data

## 1. The Initial Plan - Twitter (X) and Reddit

Our initial plan was to pull data from Twitter and Reddit. These platforms are widely regarded as the most popular online discussion platforms. They have massive, highly active forums and communities dedicated to stock market chatter.

However, during our development, we have encountered the following obstacles.

For Twitter (X), formal developer access must be requested in advance. According to online users, the new approval process is reported to take weeks if not months, affecting our timeline in completing the project. 

Similarly, for Reddit’s API, a developer app must be registered on their site to obtain the necessary credentials (Client ID and Secret), then navigate through their OAuth 2.0 setup. 

We have applied for access to both social media platforms’ APIs, but are uncertain whether and when we will be granted access, especially given the timeline of our project. 

## 2. Our Pivot - BlueSky

Considering the need to progress, we have explored alternative options and decided to pivot to BlueSky. BlueSky is an American microblogging social media service similar to Twitter, with a growing number of users and feeds focused on investing. Most importantly, BlueSky’s API is completely open and free, with no application process or waiting period. 

The API lets us grab up to 100 posts per request, along with designated parameters. Here is a quick look at the code we used to set up the API request, with an example looking into posts related to the Reddit IPO during a specified period:
```python
# === CONFIGURATION - REPLACE WITH YOUR CREDENTIALS ===
   BLUESKY_HANDLE = "xxxxxxx"
   BLUESKY_APP_PASSWORD = "xxxxxx"

# Search parameters
   QUERY = "Reddit IPO"
   START_DATE = "2024-03-01"
   END_DATE = "2024-03-21"
   MAX_POSTS = 1000

# === COLLECT POSTS ===
   posts_data = collect_posts(client, QUERY, START_DATE, END_DATE, MAX_POSTS)
   if not posts_data:
       print("No posts found matching criteria")
       return
```

## 3. Testing the Setup
Through the API, we have fetched post data containing the raw text, exact timestamp, and engagement metrics such as the number of likes. We’ve managed to compile all of this into a single CSV file.
```python
   # === SAVE CSV ===
   df = pd.DataFrame(posts_data)
   filename = f"bluesky_reddit_ipo_{START_DATE}_to_{END_DATE}.csv"
   df.to_csv(filename, index=False)
   print(f" Saved {len(df)} posts to {filename}")
```
Still, the data collected so far is messy and unclean, potentially containing duplicates and unreadable text, including Unicode and emojis. Our next step is to perform data cleaning, preprocessing, and noise filtering so that high-quality data can be provided to our NLP models and accurate, reliable sentiment scoring can be achieved.

# Conclusion and Next Steps

As of now, we have defined and selected the types of data we will need to collect and their sources. News and financial data sources will be covered in a separate blog post.

We have decided to focus on 10 IPOs and will define our selection criteria, select variables for analysis, and collect textual data, including social media posts from Blueow, as well as news and financial data. To analyse correlations and build a reliable prediction model, we will run sentiment and text analytics on the collected textual data and backtest against financial data. 

At this stage, in addition to defining the main variables and data sources, we have set up the basic codes, APIs, and methods to collect the necessary data from the appropriate sources. 

The next step will be to collect the textual and financial data for the 10 IPOs. Then, data cleaning and preprocessing will take place, where we remove duplicates and unreadable content, ensure high-quality input to our model, and validate the finalised data before running it through our model for further sentiment analysis. Simultaneously, an appropriate sentiment analysis package and model will need to be selected and developed, along with the features and code that allow us to input our cleaned data and produce meaningful outputs and insights, as well as a comparison model and method for comparing against financial data for backtesting purposes to evaluate the accuracy of our model. 
