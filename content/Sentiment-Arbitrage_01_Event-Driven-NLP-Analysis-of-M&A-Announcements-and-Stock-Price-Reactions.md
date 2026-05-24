---
Title: Event-Driven NLP Analysis of M&A Announcements and Stock Price Reactions
Date: 2026-01-15 17:12
Category: Reflective Report
Tags: Sentiment Arbitrage
Slug: sentiment-arbitrage-01-event-driven-nlp-ma
---

By Group "Sentiment Arbitrage"

## Introduction
In our NLP project, we aim to analyze how M&A announcements affect stock prices by combining announcement texts with financial data. However, we quickly ran into a major hurdle: finding reliable, high-quality data. Below is a log of our exploration and the lessons learned along the way.

---

## Part I: Sourcing M&A Announcements

### Version 1: Trying Newsdata.io

We started with **Newsdata.io**, but encountered four main problems:

1. **No full text** – Only abstracts of the announcement were provided, which was insufficient for NLP analysis.
2. **Inaccurate keyword search** – The results contained many irrelevant articles such as market forecasting, comment, rumors.
3. **Limited time window** – The free data provided is only for the most recent 48 hours, so historical analysis is not possible.
4. **Small data volume** – The total number of articles was too small for meaningful modeling.

**Attempted solution:**  
To solve problem 1 “No full Text”, we attempted to built a scraper using **Beautiful Soup** library, but it pulled noisy content (ads, `<title>` sections). Switching to **Newspaper3k** improved extraction quality, but some websites blocked scraping entirely. In total, we managed to collect only around 1,000 samples.

---

### Version 2: Moving to Newsapi.ai

We switched to **Newsapi.ai** to overcome the limitations of Newsdata.io. Luckily, Newsapi.ai provides data from the past month, and the full text of the articles can be downloaded.

**New issue:**  
However, keyword search remained imprecise. Although the dataset expands to 13k articles, most were unofficial announcements. These are mostly news reports or analyst commentaries on mergers and acquisitions, which can lead to duplication and information noise.

---


### Version 3: Exploring Hugging Face with Custom Filtering

To access older data and gain more control over filtering, we turned to **Hugging Face** datasets. This time, we successfully obtained news data spanning multiple years, and came up with a moderate data volume (~5k). While this scale was sufficient for initial exploration, the process revealed significant limitations in data quality and required substantial manual intervention in filtering.

**Cons:**  
We had to design our own multi-stage filtering logic:

1. **Dataset selection** – We shortlisted about 50 potentially relevant datasets using broad keywords including “news”, “financial news”, and “M&A”.

2. **Sample filtering** – We then scanned these datasets using transaction‑focused keywords such as “acquire”, “acquired”, and “acquisition” to extract candidate texts related to mergers and acquisitions.


**Result:**  
Even after filtering, the data remained noisy. Most retrieved texts were news discussions or commentaries on M&A events, rather than the formal, official company announcements we needed for our analysis just like version 2.

---


### Version 4: Abandoning News APIs – Moving to SEC Filings

Facing persistent noise in news‑based data, we evaluated two improved strategies:

- **Option A:** Refine Hugging Face samples using a pretrained classification model. However, this would require substantial computational resources for local inference and did not align with our immediate priorities.
- **Option B:** Source information directly from its original, official channel – the **U.S. Securities and Exchange Commission (SEC)**.

We selected **Option B** as the more reliable and efficient path.

**Current status:**
Using the **SEC API**, we collected all **DEFM14A**((merger-related proxy statement) filings since 2001 as our primary textual dataset. The resulting volume is appropriate and sufficient for our research objectives.

The following code shows a function retrieving the direct link to a **DEFM14A** from the SEC EDGAR database. Using a company's **CIK** and **accession number**, it constructs the URL for the filing's index page, fetches the HTML, and parses the document table using **BeautifulSoup**.
The script iterates through the filing's attachments to find a match for the specific "DEFM14A" document type. Once found, it converts the relative path into a full absolute URL and returns it. It also includes built-in delays to comply with SEC rate-limiting policies, ensuring a reliable and automated way to extract key legal documents from financial filings.


```python
def find_doc_from_index_page(cik, accession):


   if not cik or not accession:
       return None


   acc_nd = accession.replace("-", "")
   base_dir = f"{SEC_BASE}/Archives/edgar/data/{cik}/{acc_nd}"
   index_url = f"{base_dir}/{accession}-index.htm"


   try:
       resp = requests.get(index_url, headers=HEADERS, timeout=30)
       time.sleep(DELAY)
   except requests.RequestException:
       return None
   if resp.status_code != 200:
       return None


   soup = BeautifulSoup(resp.text, "lxml")
   table = soup.find("table", class_="tableFile")
   if not table:
       return None

   def abs_url(href):
       if href.startswith("http"):
           return href
       if href.startswith("/"):
           return f"{SEC_BASE}{href}"
       return f"{base_dir}/{href}"

   for row in table.find_all("tr")[1:]:
       cells = row.find_all("td")
       if len(cells) >= 4:
           ftype = cells[3].get_text(strip=True).upper()
           if ftype == "DEFM14A":
               link = cells[2].find("a")
               if link and link.get("href"):
                   return abs_url(link["href"])


   return None
```

Here is a sample of our downloaded news data, with tile, date, and full text.

![Picture showing Powell]({static}/images/Sentiment-Arbitrage_01_SEC-API-Output.png )

**Remaining challenge:**  
The SEC timestamps are only accurate to the day. Since we plan to study stock price reactions at a high frequency (minute or hour level), we are now working to improve timestamp granularity to align with financial market data, which could possibly solved by extracting the full **acceptance datetime** from EDGAR’s original filing metadata. This field records the exact hour, minute, and second (in Eastern Time) at which each disclosure is officially received and published by the SEC.

---


## Part II: Retrieving Financial Data

With the announcement texts and dates in hand, the next step was to retrieve stock price data for the companies involved. This follows feature engineering, where we used Named Entity Recognition (NER) to identify target companies and their corresponding announcement dates. We chose to develop our data collection method proactively to avoid last‑minute bottlenecks.

### Initial Choice: AlphaVantage

We first selected **AlphaVantage** for its stable API, which promised consistent and reliable data feeds. However, the intraday and extended historical data required for our analysis was only available through a paid tier. With no viable free access, we had to pivot.

---

### Alternative: yfinance (Yahoo Finance)

We adopted **yfinance**, an unofficial Python library that interfaces with Yahoo Finance. It proved efficient and user‑friendly for retrieving daily stock data.

**Workflow:**
```python
# --------------------------
# 2. DOWNLOAD ALL DATA
# --------------------------
all_data = []  # Store ALL stock data here

for idx, row in df_tickers.iterrows():
    ticker = row['ticker']
    start = row['start_date']
    end = row['end_date']
    
    print(f"Downloading: {ticker} | {start} to {end}")
    
    # Download stock data
    stock = yf.Ticker(ticker)
    data = stock.history(start=start, end=end, interval="1d", actions=True)
    
    # Add ticker column so you know which stock it is
    data['Ticker'] = ticker
    
    # Add to master list
    all_data.append(data)

# Combine all stocks into ONE DataFrame
final_df = pd.concat(all_data)
```

With just a few lines of code, we retrieved all essential trading metrics: opening price, daily high, daily low, closing price, and trading volume.

---

### Capabilities and Limitations of Daily Data

This approach effectively meets basic analytical needs for daily price data. When an M&A announcement is made after market close, we can assess market reaction by comparing the previous day's closing price to the next day's opening price. The gap reflects overnight market sentiment influenced by the news.

However, daily data has clear limitations:
- An announcement made **during trading hours** can trigger immediate price swings and volatility that daily data fails to capture.
- **1‑minute interval data** is crucial for monitoring rapid price changes and assessing real‑time market reactions

Unfortunately, Yahoo Finance (accessed via yfinance) imposes strict limitations on historical intraday data: 1‑minute intraday data is only available for the last 30 days, and hourly interval data is restricted to the past 730 days. Thus, for comprehensive, long‑term high‑frequency analysis, this is insufficient. Alternatives like AlphaVantage and Polygon offer more granular data, but both require paid subscriptions—a less ideal fit for our current setup.

---

## Conclusion and Next Steps

Our journey took us from commercial news APIs to open‑source datasets and finally to official regulatory filings. We have solved the core issues of data availability and authenticity. The next step is to refine the timestamp resolution. Also, we have decided to adopt **Wind** as our primary financial data platform going forward. We will begin testing it next week. With Wind, we aim to obtain high‑frequency intraday data (minute‑level) over extended historical periods and align announcement timestamps more precisely with stock price movements.

