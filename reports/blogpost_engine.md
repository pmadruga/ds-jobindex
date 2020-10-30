# Deep Learning: Applying Google's Latest Search algorithm on biggest Danish job site

## Disclaimer
This is a non-technical post. For the full code, visit <GITHUB_LINK>. For a technical explanation, get in touch <CONTACT_PAGE>.

## Abstract
Search engine results is a common cause, especially for non-English languages. In this project, different models - including Google Search's underlying technique (called BERT) - are applied to improve the search engine of Denmark's biggest job portal: jobindex.dk. The goal is to determine whether modern Artificial Intelligence and Natural Language Processing techniques can improve danish-language searches. After comparing direct results with TFIDF and BERT, it's possible to conclude that BERT excels for zero-results search queries and on longer search queries. It also provides noteworthy performance results, in terms of time to return a search. 

## Disclaimer

## Intro
Searches are a recurrent challenge, regardless of the user and/or the company. Most entities rely on direct searches of terms. Let's take for example (based on "https://blog.google/products/search/search-language-understanding-bert/") that you're searching for "2019 brazil traveler to usa need a visa.". You'd probably get a result that includes the words "2019", "brazil", "traveler", "usa" and "visa". But the word "to" would be ignore in this type of search, since it's too generic. In fact, the word "to" changes everything in this, since it refers to where the person wants to travel to (from Brazil to the US). This results is very different outcomes of the search query.


Non-english languages pose a challenge in terms of Machine Learning challenges, since there's not as much data to train models when compared to the english language. However, recent breakthroughs allows having good model results, even in Multiligual approaches. In November, 

## The dataset
