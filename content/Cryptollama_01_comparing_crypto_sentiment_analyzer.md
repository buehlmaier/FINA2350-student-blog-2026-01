---
Title: Comparing Crypto Sentiment Analyzers: A Practical Evaluation Using LLMs
Date: 2026-03-28 21:34
Category: Reflective Report
Tags: Group Cryptollama
---

By Group "Cryptollama"

As highlighted in our first presentation, the cryptocurrency market is becoming increasingly significant globally, and it shows a unique financial attribute of high correlation with market emotion. Therefore, sentiment analysis has become an indispensable tool for helping the public navigate market trends and make informed investment decisions.
However, not all sentiment analyzers are created equal. Some are rule-based, others use machine learning, and more recently large language models (LLMs) have entered the space. We built a sentiment analysis API using Ollama and FastAPI in this project.
In this blog post, we would like to demonstrate its better contextual awareness compared with traditional NLP methods.

## Methodology
We have already collected a dataset of hundreds of crypto-related posts and news articles from platforms like Reddit, CoinDesk and Cointelegraph. We fed the exactly same data to each analyzer and got their outputs of sentiment scores or more results through the following code. For evaluation of the results, both sentiment classification (bullish/bearish/neutral) and quality of reasoning are both considered.

```
with open(input_file, 'r', encoding='utf-8') as f_in:
    for line in f_in:
        if line.strip():  # Skip empty lines
            entry = json.loads(line.strip())
            text = entry.get('text_for_model', '')

            # TextBlob sentiment
            tb_polarity = TextBlob(text).sentiment.polarity
            entry_tb = entry.copy()
            entry_tb['sentiment_textblob'] = tb_polarity
            f_tb.write(json.dumps(entry_tb, ensure_ascii=False) + '\n')

            # Vader sentiment
            vader_scores = sid.polarity_scores(text)
            entry_vd = entry.copy()
            entry_vd['sentiment_vader'] = vader_scores['compound']  # Using compound score
            f_vd.write(json.dumps(entry_vd, ensure_ascii=False) + '\n')

            # Afinn sentiment
            afinn_score = afinn.score(text)
            entry_af = entry.copy()
            entry_af['sentiment_afinn'] = afinn_score
            f_af.write(json.dumps(entry_af, ensure_ascii=False) + '\n')
```

To illustrate the gap between these models, especially the advantage of LLM-based analyzers in understanding specific financial contexts compared to general-purpose tools, we picked one specific test case from our dataset:

```
{
    "source": "cointelegraph",
    "url": "https://cointelegraph.com/news/what-happens-to-bitcoin-if-oil-price-hits-180-per-barrel?utm_source=rss_feed&utm_medium=rss_category_market-analysis&utm_campaign=rss_partner_inbound",
    "published_at": "2026-03-20",
    "fetched_at": "2026-03-26T15:35:05Z",
    "title_clean": "What happens to Bitcoin if oil price hits $180 per barrel?",
    "summary_clean": "A 70% oil spike could nearly double US inflation, slash rate-cut hopes, and deepen downside risks for Bitcoin prices in the coming months.",
    "crypto": "Bitcoin",
    "text_for_model": "What happens to Bitcoin if oil price hits $180 per barrel?. A 70% oil spike could nearly double US inflation, slash rate-cut hopes, and deepen downside risks for Bitcoin prices in the coming months.."
}
```

This news describes how the recent surge in oil prices will affect the Bitcoin market. Everyone can easily tell that the news is very pessimistic. Let’s take a look at the performance of each model:

```
{
    "sentiment_textblob": 0.0
    "sentiment_afinn": -5.0
    "sentiment_vader": -0.34
}
```

Based on simple dictionary matching relying on predefined keywords, TextBlob failed completely in understanding the financial implication of the news and returned a neutral 0.0. It likely viewed words like ‘inflation’ as neutral nouns and the simple polarity lexicon couldn’t find the conditional logic between the title and the summary. 
In contrast, the two social-media-focused models, afinn and VADER, successfully identified a negative tone by catching keywords like “downside” or “slash”. While they correctly recognized a negative sentiment, they failed to weight the severity of the news, because only general possibility of “bad news” was flagged by counting negative hits rather than through financial logic. Moreover, their results remained shallow since no confidence score or explanation was given to show how the judgments were made.

Here is the output of Ollama:
```
{
    "score": -0.8, 
    "label": "bearish", 
    "confidence": 0.8,
    "explanation": "Tone indicates negative market pressure (score -0.800)."
}
```

In stark contrast, Ollama correctly identified the news as a strong bearish signal and assigned a score of -0.8. Unlike its counterparts, the LLM understands the overall meaning of expressions based on context rather than simply detecting keywords. Therefore, it didn’t just look for the number of bad words, but knew well how bad the impact of oil spikes and inflation will be on Bitcoin.
In addition to the capability of economic reasoning which traditional models lack, the LLM also provided additional confidence and explanations. These human-readable explanations are its significant and unique advantage. For financial NLP developers, this interpretability makes debugging much easier. For users in real applications, being able to explain how the results are obtained can better build trust.
To summarize our observation from this comparison:
“Traditional models detect signals, but LLMs understand narratives.”

## Conclusion, Reflection & Future Outlook
Our case study highlights a clear trend that LLM-based sentiment analysis offers a significant improvement in understanding and interpretability, especially in complex domains like crypto. By leveraging tools like Ollama and FastAPI, it is now possible to build powerful, local-first sentiment systems with minimal infrastructure, and data privacy could be better secured.
Despite its advantages, the LLM approach is not perfect. It requires much more computational resources and time, making it slower than rule-based systems and less suitable for high-frequency trading. LLM outputs can vary a lot depending on prompt design, requiring rigorous prompt fine-tuning to ensure consistent scoring. Additionally, confidence scores are often heuristic rather than statistically grounded, unlike those from trained ML models, which must be handled with caution in risk management.
For real-world applications, a hybrid approach combining speed with interpretability may be ideal. Firstly use rule-based models like VADER for rapid initial lightweight filtering, and then route high-impact or ambiguous news to LLMs for deep contextual reasoning.
Our next phase will involve manually labelling a few texts from each data source and comparing them to how the price of cryptocurrencies moved at the time the articles were published. We will use this information to iteratively improve our model by implementing more preprocessing and data cleaning steps if necessary as well as improving LLM prompting to get more optimal results. We may also fine tune our LLM since cryptocurrencies and their uses evolve rapidly, thus, our model needs to adapt accordingly as well.