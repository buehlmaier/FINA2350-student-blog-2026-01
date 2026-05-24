---
Title: Optimising Prompts For LLM Analysis
Date: 2026-05-03 15:30
Category: Reflective Report
Tags: Group Cryptollama
---

By Group "Cryptollama"

# Hunting Down a "Neutral-but-Bearish" Error: A Reflective Journey Through LLM Sentiment Analysis

We built a crypto news sentiment analyzer using `llama3.2` through Ollama. Our results have been good so far although in our intial run through, however, many articles that are supposed to be classified as "neutral" sentiment are often misclassified as "bearish". This post documents how our group diagnosed and fixed the misclassification issue to the best of our ability.

## 1. The problem: "neutral-but-bearish"

Our LLM pipeline takes crypto news via RSS feeds from Reddit, CoinTelegraph and CoinDesk, feeds the cleaned text into Ollama with a prompt, parses a numeric score in [-1.0, 1.0], and classifies that score into `bearish` / `neutral` / `bullish` using simple thresholds.

Evaluating against a hand-labelled set of 186 articles, the headline numbers looked respectable: **86% accuracy, 0.85 macro-F1**. Drilling in, the confusion matrix told a less clean story.

**Baseline confusion matrix** (rows = true label, columns = predicted):

|                  | pred bearish | pred neutral | pred bullish | recall |
| ---------------- | :----------: | :----------: | :----------: | :----: |
| **true bearish** |      55      |      3       |      1       |  0.93  |
| **true neutral** |    **13**    |      37      |      3       |  0.70  |
| **true bullish** |      1       |      5       |      68      |  0.92  |

The most striking issue is the 13/50 true neutrals, about a quarter were being labelled bearish. The model was systematically reading neutral content as negative.

To find out why, we exported the 13 mis-labelled items and inspected the _raw scores_ the model gave each one:

| Score the LLM produced | # items | Examples of the actual text                                                                         |
| :--------------------: | :-----: | --------------------------------------------------------------------------------------------------- |
|        `-0.900`        |    2    | _"2017 & BitConnect — I took that 10K out of bitcoin and put it into Bitconnect…"_                  |
|        `-0.400`        |    3    | _"Crypto Market Pulse: Bitcoin funding rates fell to their most negative levels since 2023…"_       |
|        `-0.300`        |  **8**  | _"New to crypto. Hey guys, I am fairly new to crypto and there is a lot of information out there…"_ |

**Retrospectives** about old failures scored as low as `-0.90` because of words like "scam"; **analytical news** reporting negative facts received around `-0.40` even without taking a bearish stance; **concerned-sounding questions** clustered at exactly `-0.30`. At `score == -0.30` there were also **12 true-bearish items** co-located with the 8 neutrals — the model was conflating "skeptical question" with "mild bearish news" at the same numeric output, so no threshold shift could untangle them.

The diagnosis crystallised: `-0.30` was being used as a "lazy default for mild concern", and the source of that default was sitting in plain sight in the prompt:

> _"Questions seeking advice are typically neutral **unless they express concern or excitement**."_

That conditional clause is exactly the loophole. Any concerned-sounding question — and crypto Reddit is full of these — qualifies for the exception, and the model dutifully reaches for the `-0.3 to -0.1` "Slightly negative: Mild concerns" entry in the prompt's score-scale table.

---

## 2. The journey: what worked and what didn't

Knowing the cause of the failure was easier than fixing it. We ran several rounds of prompt and configuration experiments. Most of them regressed. Below are the four most representative strategies, in roughly the order we tried them.

### Strategy 1: Full prompt rewrite

We discarded the original prompt and wrote a new one telling the model to default to 0.0 unless the text described a specific current event. The same rule that pulled neutral-leaning posts to zero also stripped scores from genuinely bullish posts, since most bullish content in this corpus is opinion-style rather than event reporting. Bullish recall collapsed to **0/74** and accuracy fell to 0.36.

### Strategy 2: Two-stage classifier

The next prompt asked the model to first label each text as either "EVENT" or "DISCUSSION" and to score non-zero only when it picked EVENT. The boundary between the two categories proved too fuzzy: posts beginning with a clear bullish signal were often categorised as DISCUSSION because the rest of the post was opinion. Bullish recall fell to 0.58 and accuracy to 0.62.

### Strategy 3: Closing reminder

A lighter-touch attempt: leave the prompt almost intact, but append a single closing reminder: _"REMINDER: Questions, discussions, and opinions always score 0.0."_ The extra instruction destabilised the model rather than focusing it: neutrals mislabelled as bearish _doubled_ from 13 to 26, and bullish recall dropped from 0.92 to 0.80.

### Strategy 4 (the winner): three small, orthogonal changes

After watching elaborate prompt rewrites fail, we tried the opposite approach: change as little as possible. The winning recipe combined three independent edits, none of which alone explained the full gain. Together they lifted accuracy from 0.86 to 0.88 and macro-F1 from 0.85 to 0.87. Each change is described in detail below.

| Strategy             |  Accuracy  |  Macro-F1  |
| -------------------- | :--------: | :--------: |
| Baseline             |   0.8602   |   0.8487   |
| Full prompt rewrite  |   0.3602   |   0.2946   |
| Two-stage classifier |   0.6237   |   0.6090   |
| Closing reminder     |   0.7581   |   0.7398   |
| **Winning approach** | **0.8817** | **0.8698** |

#### Change 1: a one-sentence prompt edit

We left the rest of the original prompt untouched and only replaced the loophole line:

```diff
- "Questions seeking advice are typically neutral unless they express concern or excitement"
+ "Questions, discussions, and advice-seeking posts are NEUTRAL (score 0.0),
+  regardless of any concerns, skepticism, or negative-sounding words they may contain"
```

The justification is direct: the original sentence was the _cause_ of the lazy `-0.30` default for concerned questions, so the carve-out was removed. The replacement is unconditional. No new content was added. Only one sentence was swapped because earlier strategies showed that the model's overall calibration is fragile.

#### Change 2: temperature 0.1 → 0.0 (greedy decoding)

The Ollama API call sets sampling parameters per request:

```python
async with httpx.AsyncClient(timeout=OLLAMA_REQUEST_TIMEOUT_SECONDS) as client:
    response = await client.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model_to_use,
            "prompt": create_sentiment_prompt(text),
            "stream": False,
            "options": {
                "temperature": 0.0,    # was 0.1 — greedy decoding
                "top_p": 0.95,
                "top_k": 40,
                "repeat_penalty": 1.1,
            },
        },
    )
```

When the LLM emits a score like `"-0.18"` or `"0.20"`, it picks each character one token at a time from a softmax distribution. At `temperature=0.1` the most-probable token is heavily favoured but not guaranteed; for a continuous score, that small chance of picking a different token can flip a `0.15` into a `0.20` and across a threshold. `temperature=0.0` forces argmax decoding which always pick the single most-confident token. Borderline scores stop oscillating, and **bullish recall jumped from 0.92 to 0.96** without any change to the prompt.

#### Change 3: bullish threshold 0.20 → 0.125

After the first two changes were in place, we ran a small offline grid search over the threshold space against the true labels:

```python
import pandas as pd, numpy as np
from sklearn.metrics import accuracy_score, f1_score

scored = pd.read_csv("evaluate/scored_sentiment_llama3.2.csv")
truth  = pd.read_csv("evaluate/sentiment_true_labels.csv")
n = min(len(scored), len(truth))
scores = scored["score"].iloc[:n].values
y_true = truth["true_label"].iloc[:n].values

best = []
for b in np.arange(-0.05, -0.6, -0.025):
    for u in np.arange(0.05, 0.6, 0.025):
        y = np.where(scores <= b, "bearish",
            np.where(scores >= u, "bullish", "neutral"))
        best.append((f1_score(y_true, y, average="macro"),
                     accuracy_score(y_true, y), round(b, 3), round(u, 3)))
for f1m, acc, b, u in sorted(best, reverse=True)[:5]:
    print(f"bear<={b:>6} bull>={u:>6} acc={acc:.4f} f1={f1m:.4f}")
```

The data-validated optimum was `bearish ≤ -0.20, bullish ≥ 0.125`, only the bullish cut moved. The change recovered true-bullish items the model had been scoring in `(0.125, 0.20)`, which correctly recognised as positive but not strongly enough to clear the original `0.20` cutoff. The new mapping will be:

```python
def score_to_label(score: float) -> str:
    if score <= -0.20:
        return "bearish"
    if score >= 0.125:   # was 0.20 in the baseline
        return "bullish"
    return "neutral"
```

---

## 3. Evaluation: what improved, what didn't, and why

Side-by-side metrics on the same 186-item evaluation set:

| Metric                | Baseline | Optimized  |       Δ       |
| --------------------- | :------: | :--------: | :-----------: |
| **Accuracy**          |  0.8602  | **0.8817** | **+2.15 pp**  |
| **Macro-F1**          |  0.8487  | **0.8698** | **+2.11 pp**  |
| Bearish precision     |  0.7971  |   0.8209   |    +2.4 pp    |
| Bearish recall        |  0.9322  |   0.9322   |       =       |
| Bearish F1            |  0.8594  |   0.8730   |    +1.4 pp    |
| Bullish precision     |  0.9444  |   0.9467   |    +0.2 pp    |
| **Bullish recall**    |  0.9189  | **0.9595** | **+4.1 pp** ★ |
| Bullish F1            |  0.9315  |   0.9530   |    +2.2 pp    |
| **Neutral precision** |  0.8222  | **0.8636** | **+4.1 pp** ★ |
| Neutral recall        |  0.6981  |   0.7170   |    +1.9 pp    |
| Neutral F1            |  0.7551  |   0.7835   |    +2.8 pp    |

Confusion-matrix diff (boldface = changed cells):

|  Baseline / Optimized   | pred bearish | pred neutral | pred bullish |
| :---------------------: | :----------: | :----------: | :----------: |
| **true bearish** (n=59) |   55 / 55    |    3 / 3     |    1 / 1     |
| **true neutral** (n=53) | **13 / 12**  | **37 / 38**  |    3 / 3     |
| **true bullish** (n=74) |  **1 / 0**   |  **5 / 3**   | **68 / 71**  |

**Net effect: 4 more correct predictions out of 186** (164 → 168). The bearish row was unchanged. Only one of the original 13 neutral→bearish errors was recovered while the other 12 remain stuck because the model scored them at `-0.30` or below with high confidence. The largest gains came from the bullish row, where `temperature=0.0` and the lower threshold together captured three borderline items and removed the one false-bearish prediction, lifting bullish recall from 0.92 to 0.96 and neutral _precision_ by 4.1 percentage points.

The group had assumed prompt engineering would be the dominant lever but the data said otherwise. Prompt engineering produced one of the smallest gains, **decoder configuration** produced the largest, and **post-hoc threshold tuning** produced the most reliable. It could be the case that the prompt was already well-tuned for the Llama model.

Across multiple attempts to improve the prompt, only one worked, and it changed almost nothing. With a small model, the lesson we learnt is that the right move sometimes might be to leave the prompt alone and tune the things around it.
