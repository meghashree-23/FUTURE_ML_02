# FUTURE_ML_02
# Support Ticket Classification & Prioritization
**Future Interns – Machine Learning Task 2 (2026)**

## Objective
Build a machine learning system that reads a customer support ticket, assigns it to a category, and gives it a priority level (High / Medium / Low), so support teams spend less time sorting and more time solving.
## Summary for Business Readers
**The problem:** support teams lose time sorting tickets by hand, and urgent problems can get buried.

**What this system does:** it reads a ticket, sends it to the right category (Access, Hardware, HR Support and so on), and flags how urgent it is (High, Medium or Low).

**How well it works:** it picks the right category for about 85 out of 100 tickets. Guessing the most common category would be right about 28 times out of 100.

**What it means for operations:** most tickets can be routed automatically, urgent tickets (about 7% of the total) are flagged first, and staff time goes to solving problems instead of sorting them. Tickets about administrative rights are the hardest for the model, so a person should still review those.

## Repository Contents
| File | What it is |
| `Task2.ipynb` | Full notebook: cleaning, training, evaluation, priority rules, demo |
| `ticket_category_model.joblib` | Saved category model |
| `tfidf_vectorizer.joblib` | Saved text-to-numbers converter (needed to use the model) |
| `all_tickets_processed_improved_v3.csv` | Dataset used |
| `t2_*.png` | Charts and demo screenshot |
| `requirements.txt` | Python libraries needed |

## Dataset
**IT Service Ticket Classification Dataset (Kaggle)**: 47,837 real help-desk tickets with the ticket text (`Document`) and its category (`Topic_group`). After cleaning, 47,835 tickets were used (2 became empty and were removed).

Categories: Access, Administrative rights, HR Support, Hardware, Internal Project, Miscellaneous, Purchase, Storage.

**Dataset check:** I first tested the Kaggle *Customer Support Ticket Dataset*. A TF-IDF + Logistic Regression model scored 21.1% on category (guessing the biggest class: 20.7%) and 46.2% on priority (guessing: 49.8%). The text had no learnable link to the labels (templated descriptions, near-perfectly balanced classes), so I switched to the dataset above, which has real labelled text. The first dataset's CSV is not included in this repository; the test results are in the notebook outputs.

## How Tickets Are Categorized
1. **Text cleaning:** lowercase, remove punctuation and numbers, remove English stopwords (NLTK) and very short words.
2. **Feature extraction:** TF-IDF with 5,000 features, using single words and word pairs.
3. **Model:** Logistic Regression, trained on 80% of tickets and tested on a held-out 20% (stratified split).

## How Priority Is Decided
The dataset has no priority labels, so priority is assigned by **transparent, rule-based scoring**:

| Rule | Points |
| Contains an urgent word (e.g. urgent, outage, down, crashed, breach, virus) | +3 |
| Contains a problem word (e.g. error, failed, unable, locked, broken) | +1 |
| Predicted category is Access, Hardware or Storage (work is usually blocked) | +1 |

**3 or more points = High, 1–2 = Medium, 0 = Low.**

Note: these rules were designed by me, not learned from data, so there is no accuracy score for priority.

Resulting distribution: High = 3,295 (6.9%), Medium = 27,095 (56.6%), Low = 17,445 (36.5%).

## Results (Category Model)
| Metric | Value |
| Accuracy | **85.3%** |
| Baseline (always guess the biggest class) | 28.5% |
| Macro-average F1 | 0.85 |

Strongest categories: Purchase (precision 0.97), Access (F1 0.90), Storage (F1 0.88).
Weakest: **Administrative rights** (recall 0.60). 107 of its 352 test tickets were predicted as Hardware.

### Visualizations
Category counts(t2_01_category_counts.png)
Confusion matrix (t2_02_confusion_matrix.png)
Priority distribution (t2_03_priority_distribution.png)

### Dem0
Demo (t2_04_demo.png)

## Insights for a Support Team
- **Faster routing:** about 85% of tickets can be sent to the right team automatically, saving manual sorting time.
- **Urgent issues surface first:** tickets with words like "down" or "outage" are flagged High, so they are not buried in the queue.
- **Where humans are still needed:** Administrative rights tickets are often confused with Hardware, so these should be reviewed by a person.
- **Workload planning:** Hardware and HR Support are the biggest categories, which helps with staffing.

## Limitations
- Priority uses hand-written rules and has not been validated against real priority labels.
- The model leans toward Hardware, the largest class, when ticket wording is unclear (for example, "buy a new monitor" was labelled Hardware instead of Purchase).
- Small categories such as Administrative rights have fewer examples and lower recall.
- Keyword lists are English only and may miss urgent tickets worded differently.

## Future Improvements
- Learn priority from real labelled data instead of rules.
- Try stronger models (Linear SVM, XGBoost, or transformer models such as BERT).
- Handle class imbalance (class weights) to improve recall on small categories.

## Tools Used
Python, Pandas, NLTK, Scikit-learn, Matplotlib, Google Colab
