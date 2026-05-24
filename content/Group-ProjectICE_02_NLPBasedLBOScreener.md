Title: Progression reflection on our NLP model by Group Project ICE
Date: 2026-05-03  
Category: Final Report
Tags: Group Project ICE

This post details the latest updates from Group Project ICE.

**Financial data cleaning**

Following our plan from the first blog post, we resolved the S\&P500 financial data cleaning issues regarding missing information in two different ways accordingly to our suggested plan. First, when the primary label such as cost of revenue is missing, the model will look for established alternative field names, such as cost of goods sold or cost of sales. Second, the model will also derive missing information through accounting formulas, such as deducing gross profit by subtracting cost of revenue from revenue. Collectively, these two implementations successfully address field name inconsistency data collection, reflecting the effectiveness of our initial plan, or so we thought.

```python  
gross\_profit \= g\_income(col, "Gross Profit", "Gross Income")  
            if gross\_profit is None:  
                cost\_of\_revenue \= g\_income(  
                    col,  
                    "Cost Of Revenue", "Cost of Revenue",  
                    "Reconciled Cost Of Revenue", "Cost Of Goods Sold",  
                    "Cost Of Sales", "Reconciled Cost Of Goods Sold"  
                )  
                if revenue is not None and cost\_of\_revenue is not None:  
                    gross\_profit \= revenue \- cost\_of\_revenue  
```  
   

While we found a great improvement in data cleaning, we still encountered a large range of missing field after data cleaning, which became very confusing. However, we eventually realized that this only mostly happened within financial sector firms with the contributing factor eventually identified as their structural differences. Since financial firms operate differently from the typical industrial company, which buys inputs and sells outputs while funding operations with borrowings, they do not report items like free cash flow and capital expenditure, as their cash flows are driven by flow of loans and deposits rather than capital expenditure cycles, where companies invest money in assets to increase production and generate cash flow. Given that our screening criteria depends heavily on free cash flow, financial sector companies were breaking our pipeline structurally due to information that never existed beyond mere inconsistency in field names as we previously thought. Therefore, it was determined that the most appropriate resolution was to exclude them from the model’s candidate pool entirely, done through the below codes. 

```python  
if sector \!= "Financials":  
    tickers.append(ticker)  
else:  
    excluded.append(ticker)  
```

This decision is further supported by the general infeasibility of financial sector firms as LBO candidates. Not only are they already heavily leveraged by nature, but their assets are predominantly financial instruments like loans rather than operational assets that can generate predictable cash flow, and regulatory capital requirements also make post-acqutision restructuring severely constrained. As a result, financial sector firms were excluded from the model screening as due to both technical and financial reasons. Reflecting on this decision, we believe this was the right call as data completeness improved significantly, with only occasion missing fields that can be neglected in impact. 

What we learnt from this experience was more than just technical coding issues. As we originally assumed that data collection would be relatively simple through simple lines of coding, but we learned that real world financial data is inconsistent and does not always conform to a single structure as well due to the diverse ways of business operations, illustrating the difficulty behind data standardization beyond technical coding.

These 2 graphs illustrate the before and after data completeness, with empty fields in red

![]({static}/images/Group-ProjectICE_02_figure1.png)
**Figure 1 \-** Before data completeness (I)

![]({static}/images/Group-ProjectICE_02_figure2.png)
**Figure 2 \-** Before data completeness (II)

![]({static}/images/Group-ProjectICE_02_figure3.png)
**Figure 3 \-** After data completeness

## **Merging the NLP and Financial Layers**

Another major update was combining the NLP output with the financial dataset. This was necessary because the project was never meant to be only a text model or only a financial ratio screen. The goal was to identify companies that show signs of strategic stagnation in their filings, while also having financial characteristics that make them worth reviewing as possible LBO candidates.

At this stage, the project had two separate layers. The first layer was the NLP layer which contained company-level stagnation metrics extracted from filings, including innovation decay, strategic decay, and topic rigidity. These variables were designed to capture whether a company’s language was becoming less dynamic over time. Another layer was the financial layer which contained structured financial metrics such as interest coverage, debt-to-EBITDA, and free cash flow. These variables were not the main signal of the project, but they were necessary because an LBO candidate must be financially feasible. A company can look interesting from a language perspective, but if it cannot support debt or generate cash flow, then it is not very useful as an LBO candidate. 

The problem was that these two datasets were created separately. The NLP file was organized around filing-level or company-level textual outputs, while the financial file was organized around accounting variables and financial years. Before producing a final ranking, we had to make both layers compatible. 

The most important step was standardizing ticker symbols. If one dataset has `“aapl "` and the other has `"AAPL"`, Python treats them as different values. That type of small formatting issue can cause companies to disappear during the merge. So before combining the datasets, we stripped whitespace and converted all tickers to uppercase.

```py
# Standardize ticker symbols in both datasets
nlp_df["ticker"] = nlp_df["ticker"].str.strip().str.upper()
fin_df["ticker"] = fin_df["ticker"].str.strip().str.upper()
```

We also had to make sure the financial dataset had only one row per company. Since financial data can contain multiple years for the same ticker, we kept the latest available financial year for each company. This allowed the final ranking to compare one NLP stagnation score with one financial feasibility score per company.

```py
# Keep the latest financial year for each company
fin_latest = (
    fin_df
    .sort_values(["ticker", "year"])
    .groupby("ticker", as_index=False)
    .tail(1)
)
```

After that, we merged the NLP layer with the latest financial layer using ticker as the common identifier.

```py
# Merge NLP stagnation metrics with financial feasibility metrics
merged_df = nlp_df.merge(
    fin_latest,
    on="ticker",
    how="inner"
)

print("NLP companies:", len(nlp_df))
print("Financial companies:", len(fin_latest))
print("Merged companies:", len(merged_df))
```

We used an inner merge because we only wanted companies that had both usable NLP data and usable financial data. This reduced the final universe, but it made the results cleaner. In our larger run, the NLP file had more than 300 companies, but after merging with the clean financial dataset, the final ranking universe became 119 companies. We initially thought this was a failure of the model as we assumed that a good model should be able to evaluate as many companies as possible. However, we learned that the excluded companies were dropped because they lacked usable NLP or financial data to be properly evaluated, and including them in the final ranking would have only produced quantitative output that has no practical qualitative information. So rather than this being a failure, we understood that it was a clean overlap between the textual and financial layers instead.

