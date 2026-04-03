Title: Starting with Language: Our Early Shift in Building an LBO Candidate Model (by Group "Project ICE")
Date: 2026-03-28
Category: Reflective Report
Tags: Group Project ICE

# Starting with Language: Our Early Shift in Building an LBO Candidate Model

When we first planned this project, the idea seemed pretty clear. We wanted to build a model that could identify potential LBO candidates by combining two things: financial feasibility and text-based signals of strategic stagnation. The logic was to split it evenly, approximately a 50/50 weighted score. The finance side would tell us whether a company could realistically support leverage, and the NLP side would tell us whether the company seemed strategically stagnant before the financial metrics show it.

However, as we kept developing the project, we realized that approach probably was not the best fit for the class or for what makes the project genuinely interesting. Since the course puts a strong emphasis on textual analysis, it started to feel like making NLP just one half of the model was underselling the most distinctive part of what we were doing. So we made a shift in the project’s direction by starting with the language itself and building the stagnation signals first instead.


![Initial project direction]({static}/images/Group-ProjectICE_01_figure1.png)

**Figure 1:** Our initial project direction

![New project direction]({static}/images/Group-ProjectICE_01_figure2.png)

**Figure 2:** Our new project direction

So right now, the front end of the project is fully NLP-centered. The financial side still matters and will absolutely come in later, but at this stage we are first trying to answer a more focused question: Can a company’s language reveal strategic stagnation before the numbers fully do? 

That shift makes sense academically, but it also creates an obvious practical tension. If we begin with text first, we might spend time analyzing companies that are not even good LBO candidates financially. A company may sound stagnant in its disclosures and still be too levered or too unstable to take private. So from an actual deal or financial perspective, an NLP-first process is not the most efficient, but from a research perspective, it is useful because it lets us isolate whether language itself has anything to contribute to identifying strategic stagnation within firms. Frankly, that tradeoff has become one of the most interesting parts of the project as we are researching whether our linguistic metrics, constructed without any knowledge of whether a buyout occurred, are statistically associated with eventual acquisition.

## Operationalizing "Stagnation": Easier Said Than Done

Once we committed to NLP-first, the next question is: What does strategic stagnation actually look like in text?

This was actually way harder than we thought. Our first instinct was to utilize the Management Confidence Signal by looking for negative or uncertain language in MD&A, particularly pessimistic words and hedges. However, we realized that a company which says *"we face significant headwinds but have launched three new product lines to address them"* is using uncertain language while being strategically active. Meanwhile, a company that is confident and suggesting strategies but has a high textual similarity across annual reports, with no new vocabulary, decreasing specificity and repeats the same strategic priorities word year after year, is in fact stagnating.

Our first instinct came directly from existing literature, particularly Price et al. (2012), who found that uncertain and negative tone in earnings calls predicted future stock returns. We naturally assumed the same logic would apply to MD&A text. However, earnings calls are spontaneous and unfiltered while MD&A goes through weeks of legal review before filing. By the time it reaches EDGAR, genuine uncertainty has already been replaced with carefully worded neutral language. As a result, negativity and uncertainty may not accurately reflect a stagnating company, but rather vagueness in language.

Realizing that pessimistic words or uncertainty in MD&A do not necessarily signify stagnation, we had to change our approach. We moved from focusing solely on negative and uncertain language to a more advanced filter by separating **hollow confidence** (seemingly confident language that contains no concrete information and commits to no specific actions) from **genuine strategic decisiveness** (commitment to actual strategic actions).

To make the idea practical, we aim to build two lists of phrases: one capturing hollow confidence markers and one capturing genuine strategic decisiveness (specific capital allocations, named initiatives, and measurable targets with timelines). Once these lists are established, we aim to track how the ratio between the two indicators shifts within the same company over time. A firm whose MD&A is gradually replacing specific commitments with vague reassurances is exhibiting the strategic stagnation we aim to use as a signal for LBO candidacy.

| Hollow confidence                              | Genuine strategic decisiveness                                      |
|------------------------------------------------|---------------------------------------------------------------------|
| “We remain well-positioned to capture growth opportunities” | “We launched three new product lines targeting the mid-market”     |
| “We are committed to our strategic priorities” | “We allocated $200M to R&D, up 18% year-on-year”                   |
| “We are committed to delivering long-term shareholder value” | “We will invest $500M in Southeast Asia operations by 2026”       |

*Table 1: Sample list of phrases in each indicator*

## Technical Journey

While we changed our direction to research in the NLP component before taking into consideration  the financial feasibility as mentioned above, we had built a financial data collection model using **yfinance** prior to that already. This is a function that pulls annual financial statement data for a given ticker and tries to construct key metrics such as EBITDA and free cash flow.

```python
import yfinance as yf

stock = yf.Ticker(ticker)

# ----- Get financial statements -----
income = stock.financials
balance = stock.balance_sheet
cashflow = stock.cashflow
```

However, one major issue we ran into was inconsistent field names. When our prototype tried to reference specific financial metrics like total debt or free cash flow, the script would sometimes fail because yfinance returned those fields under alternative label names depending on the company’s choice of words. While one company may report it as "Total Debt", another may split into "Long Term Debt" and "Current Debt" separately. Prior to experiencing this issue, we had assumed that pulling data from a single standardized source would be the straightforward easy part of the project. However, we learnt the hard way that financial data in the real world is not clean or standardized. Even when it comes from a single source like yfinance, different companies report on different fiscal year schedules, use different line item names, and sometimes omit fields entirely. 

To combat this issue, we first went for a simple approach to detect whether a field actually exists in the company’s report, if not then the code should skip it and record that it is missing, rather than crashing entirely. However, we aim to clean up the data even better and in a more standardized way by building a label mapping dictionary that attempts to search for alternative field names when the expected one is missing. For example, if "Total Debt" is not found, the code would automatically look for  "Long Term Debt" and "Short Term Debt" and sum them together. We intend to refine our model to be able to handle the inconsistencies across large samples of companies through a generalized cleaning process. It is crucial that our model be able to extract financial metrics consistently as our financial feasibility stage depends entirely on how consistently we can extract metrics across companies. If key fields like total debt or free cash flow are mistreated or mislabeled, those companies may be incorrectly screened, thus undermining our model before the NLP analysis even began.


One thing that became much clearer as we worked through this is that text selection matters a lot. We are not using 10-K MD&A sections just because they are convenient to scrape. We are using them because they sit right in the middle of regulation, strategy, and comparability. Companies are required to produce them consistently, they usually follow a recognizable structure, and they include both explanations of what happened and signals about where management says the company is going. 

Later on, we still want to expand into earnings call transcripts because we believe they may capture tone and pressure more naturally than annual filings. But for now, 10-K MD&A gives us a much cleaner starting point.

Below shows how 10-K MD&A sections are extracted. 


### Extracting 10-K Filings

```python
from edgar import *

# Extract 10-K filings in the last n years
company = Company(ticker)
filings_n = company.get_filings(form="10-K").latest(n)
```

### Getting HTML Content

```python
from bs4 import BeautifulSoup
import requests

url = filing.filing_url
response = requests.get(url, headers=headers)
html_content = response.text
soup = BeautifulSoup(html_content, 'html.parser')
text = soup.get_text(separator='\n')
```

### Extracting the MD&A Section

```python
start_patterns = [
    r"item[\s]*7\.?[ \t]*management's",
    r"item[\s]*7\.?[ \t]*-[ \t]*management's",
]

end_patterns = [
    r"item[\s]*7a\.?[ \t]*quantitative",
    r"item[\s]*8\.?[ \t]*financial\s+statements",
    r"item[\s]*8\.?[ \t]*-[ \t]*financial\s+statements"
]
```

But then, we realized that the program returned “None”, suggesting it could not find the pattern. After careful investigation, we realized that some symbols in the text collected did not follow the ASCII standard encoding that our pattern matching assumed. For instance, non-breaking spaces that look identical to regular spaces were breaking our pattern matching. So, we wrote a function to normalize the text to standardize these characters and facilitate the pattern finding above and text processing. It was of significant importance that we ensure our NLP analysis runs on non-corrupted input so that the model does not reflect a broken extraction rather than a genuine linguistic stagnation signal. Therefore, even a minor bug like this was crucially important to fix in order to lay a foundation for the validity of our entire approach. This is when we learnt that text cleaning was not an initial one-time preprocessing step, but rather an ongoing process of discovering issues that only reveal themselves when the code fails. 

```python
# Normalize text for reliable pattern matching
text = text.replace('\u00A0', ' ')                    # non-breaking spaces
text = text.replace('’', "'").replace('‘', "'").replace('`', "'")
```


## Conclusion

At this point, the project is in an interesting place. We are no longer  simply trying to build a balanced model combining financial feasibility and NLP analysis to identify LBO candidates. Instead, we have shifted to an NLP-first approach by  prioritizing the research question of whether corporate language can reveal strategic stagnation before financial metrics do.

Along the way, we encountered difficulties in conceptually setting up stagnation metrics and technical coding issues. However, reflecting on these challenges has made us realise that building a model is rarely straightforward and we aim to overcome these challenges. Once we are confident the stagnation signals are reliable, we will reintegrate the financial feasibility layer to produce a complete LBO screening model. 


**References**

Price, S. M., Doran, J. S., Peterson, D. R., & Bliss, B. A. (2012). Earnings conference calls and stock returns: The incremental informativeness of textual tone. *Journal of Banking & Finance*, 36(4), 992–1011. https://doi.org/10.1016/j.jbankfin.2011.10.013
```

