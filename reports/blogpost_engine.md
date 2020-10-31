# Deep Learning: Applying Google's Latest Search algorithm on biggest Danish job site

## Disclaimer
This is a non-technical post. For the full code, visit <GITHUB_LINK>. For a technical explanation, get in touch <CONTACT_PAGE>.

## Abstract
Search engine results is a common cause, especially for non-English languages. In this project, different models - including Google Search's underlying technique (called BERT) - are applied to improve the search engine of Denmark's biggest job portal: jobindex.dk. The goal is to determine whether modern Artificial Intelligence and Natural Language Processing techniques can improve danish-language searches. After comparing direct results with TFIDF and BERT, it's possible to conclude that BERT excels for zero-results search queries and on longer search queries. It also provides noteworthy performance results, in terms of time to return a search. 

## Disclaimer

## Intro
Searches are a recurrent challenge, regardless of the user and/or the company. Most entities rely on direct searches of terms. If you open your e-mail application and search for a specific text, you get exactly what you searched for. But not all searches are black and white: sometimes you want similar results too. 

>  You shouldn't need perfect search skills to find what you want.

When performing a direct search there's no relation between words: the words in the search query will have to be present in the results, almost directly. As an example: a job search using the query "psychiatrist" would return results with that same word. But you'd probably want to see results for jobs such "doctor in psychiatry hospital". The main word here is "psychiatry" and "psychiatrist". And the chances of finding both with one query are low. What about doing long queries? The chances of getting worthy results with a direct search are very low - the longer the query, the lower the chances of success.

However, recent advances in Natural Language Processing (NLP) techniques have allowed for transforming words into a common base form. For example, words like "am", "are", "is" have their common form is the word "be" (a technique known as "Stemming"). They can also convert the plural of a word to it's singular (known as "Lemmatization"). Here's an example with both techniques: "The boys are eating pizza", after being processed with the above techniques becomes "The boy is eat pizza". Applying these two techniques to documents will definitely improve search results, finding similar results to our original search query. And those techniques are part of a few others used in this project, for a mostly Danish language dataset. This project benefits from how Danish NLP techniques have reached a very good level, when compared to English language-based ones (typically more complete).

The relations between words are also important. Let's take for example (based on "https://blog.google/products/search/search-language-understanding-bert/") that you're searching for "2019 brazil traveler to usa need a visa.". You'd probably get a result that includes the words "2019", "brazil", "traveler", "usa" and "visa". But the word "to" would be ignore in this type of search, since it's too generic. In fact, the word "to" changes everything in this, since it refers to where the person wants to travel to (from Brazil to the US). This results in very different outcomes of the search query.

But there are more challenges to this then word relations.

The above shows english language examples. Non-english languages pose a challenge in terms of Machine Learning challenges, since there's not as much data to train models when compared to the english language. However, recent breakthroughs allows having good model results, even in multilingual approaches. That's where BERT comes in. BERT is "a neural network-based technique for natural language processing (NLP) pre-training called Bidirectional Encoder Representations from Transformers" (https://blog.google/products/search/search-language-understanding-bert/). This model allows for contextual representation of words. In other words, context matters. Remember the travelling example from above.

> The above techniques set the foundations for this project, since they are part of a toolset to make search engine results better using document similarity.


## Methodology
### The dataset

The dataset is comprised of 4.2m jobs from jobindex.dk, from 2000 until March 2020. On the of the techniques used later on required using a smaller dataset. 


### The hardware

The computations were made in a Macbook Pro 2016 model.

### Preprocessing


### 
