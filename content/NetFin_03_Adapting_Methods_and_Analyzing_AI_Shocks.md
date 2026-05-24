---
Title: Unveiling Our Results- Adapting Methods and Analyzing AI Shocks
Date: 2026-05-01 19:21
Category: Reflective Report
Tags: AI, NVDA, Event Study, Sentiment Analysis
Author: NetFin Group
Slug: netfin-03-nvda-results-progress
Summary: An update on our event study progress, showcasing how we adapted our NLP methodologies across different shock categories and what our empirical results actually tell us.
---

By Group NetFin

# Unveiling Our Results: Adapting Methods and Analyzing AI Shocks

# 1. Recap

In our previous post, we detailed the unexpectedly complex process of cleaning AI news data and properly defining "T=0" for our NVIDIA (NVDA) event study. With a clean, defensible timeline established, we finally moved on to the core of our project: running the regressions to see how AI-related news sentiment actually impacts NVIDIA's Cumulative Abnormal Returns (CAR).

Before detailing our methodological adaptations, we briefly recap our **news taxonomy**. As established in our previous post, we classified AI-related news affecting NVIDIA into four distinct shock categories based on their underlying economic mechanism. **Category A (Capability Shocks)** captures major AI model launches from competitors like OpenAI and Google, which may signal shifts in GPU demand. **Category B (Hardware Architecture Shocks)** focuses on breakthrough announcements from NVIDIA's direct competitors (AMD, Intel), representing threats to NVIDIA's core market share. **Category C (Policy/Regulatory Shocks)** includes export bans and geopolitical regulations that affect NVIDIA's operational landscape. Finally, **Category D (Corporate Landscape Shocks)** comprises NVIDIA's own earnings calls, where management sentiment directly influences investor expectations. This taxonomy proved essential because each category required fundamentally different data sources and NLP approaches.

# 2. The Problems We Encounter and our Solutions

Originally, we designed a uniform pipeline: fetch all our curated events from the Alpha Vantage API, calculate a sentiment score using a single NLP model or using `nvda_sentiment_score` from the API directly, and run the regressions across the board. However, we quickly learned that treating all AI news equally is inappropriate.

A major roadblock we encountered was the **conflicting sentiment problem**: for Category B, including both NVIDIA chip launches and competitor breakthroughs meant that positive news for NVIDIA and positive news for AMD had completely opposite effects on NVDA's stock price. We solved this by narrowing the category exclusively to competitor news. Similarly, the **relevance threshold problem** forced us to abandon uniform API pulls—strict relevance filters would capture regulatory news but miss major AI model launches from OpenAI or Google. **An additional issue emerged specifically in Category A (Capability Shocks)**: when filtering for the top 50 most relevant news articles containing "NVIDIA," the API frequently returned opinion pieces and post-hoc analysis published days after the actual AI model launch event. For example, we would pull headlines like "NVIDIA Price Plummeted After DeepSeek R1 Announcement" where the news publication date lagged the actual event by 3-5 days, causing a complete misalignment between our T=0 event date and the true market-moving moment. This introduced look-ahead bias and contaminated our event window. Translating raw, unstructured text into meaningful financial insights required us to rethink our initial assumptions.

## 3. A Methodological Shift: Why Each Category Needed Separate Methods

Because of these inherent contradictions, each category needed separate methods—including cases where we did **not** use `nvda_sentiment` at all. We adapted our news category classification and methodologies as follows:

| Category | Focus | Data Source | NLP Model | Uses `nvda_sentiment`? |
|:---:|:---:|:---:|:---:|:---:|
| **A: Capability Shocks** | AI model launches (OpenAI, Google, etc.) | Manually selected | FinBERT | No |
| **B: Hardware Shocks** | NVDA competitors (AMD, Intel news) | API fetch (`nvda_competitors`) | VADER | No |
| **C: Regulatory Shocks** | Export bans, geopolitical policies | Manually selected | FinBERT | Yes |
| **D: Corporate Landscape** | NVDA earnings calls | API fetch (NVDA earnings) | FinBERT | Yes |

**Why this matters:** For Categories A and B, we are not analyzing sentiment *about* NVIDIA directly—we are analyzing sentiment about external events (competitor launches, rival hardware, etc.) and measuring how those exogenous shocks affect NVIDIA's stock. Only Categories C and D uses direct NVIDIA-specific sentiment.

Thus, we also revised our **project workflow**:

![Figure 1: Project workflow diagram showing data collection, sentiment analysis, CAR calculation, and regression interpretation.]({static}/images/NetFin_03_Workflow.png)

*Figure 1: NetFin Project Workflow. This diagram illustrates our end-to-end methodology, from fetching news through four distinct categories to final statistical interpretation.*

## 4. CAR Calculation Methodology

Before diving into the regressions, it is worth noting a crucial technical step we took regarding our data hygiene. Because we were fetching both financial price data and real-time news articles, any discrepancy in time zones could easily map an event to the wrong trading day, which would completely misalign our T=0 calculations. To ensure absolute consistency and accurately reflect the U.S. market's perspective, we routed all our API and data fetching requests through a VPN connected to a New York server. This guaranteed that every timestamp for both news publication and market pricing was cleanly synced to Eastern Time.

To measure the market impact of each news event, we calculated Cumulative Abnormal Returns (CAR) across four windows: pre-event (CAR -1), immediate shock (CAR 0), intermediate digestion (CAR 1), and incorporation (CAR 2). For beta estimation, we used a single universal NVDA beta estimated from the past 5 years of data and reused that same beta for all events under a market model specification. Below is a pseudocode summary of our implementation:

```python
FUNCTION load_price_series(path, close_name):
    df = read_csv(path)
    df['Date'] = to_datetime(df['Date'])
    df[close_name] = to_numeric(df['Close'])
    RETURN df[['Date', close_name]]

FUNCTION get_event_date(news_timestamp):
    # Map news after 4pm ET to next trading day
    IF news_timestamp.time >= 16:00:
        RETURN news_timestamp.date + 1 day
    RETURN news_timestamp.date

FUNCTION estimate_market_beta(returns_df):
    hist = returns_df   # use the full 5-year historical sample
    r_stock = hist['r_nvda']
    r_market = hist['r_qqq']
    cov = covariance(r_stock, r_market)
    market_var = variance(r_market)
    RETURN cov / market_var

FUNCTION abnormal_return(r_nvda, r_qqq, beta):
    expected_return = beta * r_qqq
    RETURN r_nvda - expected_return

FUNCTION compute_car(returns_df, event_idx, beta):
    ar = []
    FOR day in [event_idx-1, event_idx, event_idx+1, event_idx+2]:
        r_nvda = returns_df.loc[day, 'r_nvda']
        r_qqq = returns_df.loc[day, 'r_qqq']
        ar.append(abnormal_return(r_nvda, r_qqq, beta))
    
    car_m1 = ar[0]
    car_0 = ar[0] + ar[1]
    car_1 = ar[0] + ar[1] + ar[2]
    car_2 = sum(ar)
    RETURN (car_m1, car_0, car_1, car_2)

MAIN:
    nvda = load_price_series('NVDA_5Y.csv', 'Close_NVDA')
    qqq = load_price_series('QQQ_5Y.csv', 'Close_QQQ')
    merged = merge(nvda, qqq, on='Date')
    merged['r_nvda'] = pct_change(merged['Close_NVDA'])
    merged['r_qqq'] = pct_change(merged['Close_QQQ'])
    merged = merged.dropna()
    
    beta = estimate_market_beta(merged)

    FOR each news event:
        event_date = get_event_date(news.publish_date)
        t_idx = index_of(merged['Date'], event_date)
        IF t_idx is None or t_idx < 1: CONTINUE
        (car_m1, car_0, car_1, car_2) = compute_car(merged, t_idx, beta)
        OUTPUT (car_m1, car_0, car_1, car_2, news.sentiment)
```

**Key technical details:** We estimate a single universal market-model beta for NVDA using the past 5 years of return data and use the QQQ ETF as our market proxy to filter out AI-irrelevant market movements. News published after 4:00 PM Eastern Time is mapped to the next trading day to ensure proper alignment with market pricing.


## Conclusion

Looking back at the methodologies and analysis of our project, we evolved from a basic, uniform API pull to a highly tailored approach where we match specific NLP tools (FinBERT vs. VADER) to the specific contextual nature of the news, recognizing that each shock category demands its own bespoke methodology.

More importantly, working through the regressions and receiving feedback has taught us how to interpret our findings responsibly. The regression analysis underscored that different news categories impact NVIDIA's stock price in vastly different ways, and that we must always distinguish between causal narratives and simple correlation. 