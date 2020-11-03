# Deep Learning: Applying Google's Latest Search algorithm on biggest Danish job site

## Disclaimer
--- 
This is a non-technical post however some terms are impossible to escape. For the full code, visit <GITHUB_LINK>. Some terms are not detailed in depth since it that falls outside the core goal of this project. For a technical explanation, get in touch <CONTACT_PAGE>. This report also envolves knowledge of basic Danish.

## Abstract
---
Search engine results is a common cause, especially for non-English languages. In this project, different models - including Google Search's underlying technique (called BERT) - are applied to improve the search engine of Denmark's biggest job portal: jobindex.dk. The goal is to determine whether modern Artificial Intelligence and Natural Language Processing techniques can improve danish-language searches. After comparing direct results with TFIDF and BERT, it's possible to conclude that BERT excels for zero-results search queries and on longer search queries. It also provides noteworthy performance results, in terms of time to return a search. 


## Intro
---

Searches are a recurrent challenge, regardless of the user and/or the company. Most entities rely on direct searches of terms. If you open your e-mail application and search for a specific text, you get exactly what you searched for. But not all searches are black and white: sometimes you want similar results too. 

>  You shouldn't need perfect search skills to find what you want.

When performing a direct search there's no relation between words: the words in the search query will have to be present in the results, almost directly. As an example: a job search using the query "psychiatrist" would return results with that same word. But you'd probably want to see results for jobs such "doctor in psychiatry hospital". The main word here is "psychiatry" and "psychiatrist". And the chances of finding both with one query are low. What about doing long queries? The chances of getting worthy results with a direct search are very low - the longer the query, the lower the chances of success.

However, recent advances in Natural Language Processing (NLP) techniques have allowed for transforming words into a common base form. For example, words like "am", "are", "is" have their common form is the word "be" (a technique known as "Stemming"). They can also convert the plural of a word to it's singular (known as "Lemmatization"). Here's an example with both techniques: "The boys are eating pizza", after being processed with the above techniques becomes "The boy is eat pizza". Applying these two techniques to documents will definitely improve search results, finding similar results to our original search query. And those techniques are part of a few others used in this project, for a mostly Danish language dataset. This project benefits from how

>  Danish NLP techniques have reached a very good level, 

when compared to English language-based ones (typically more complete).

The relations between words are also important. Let's take for example (based on "https://blog.google/products/search/search-language-understanding-bert/") that you're searching for "2019 brazil traveler to usa need a visa.". You'd probably get a result that includes the words "2019", "brazil", "traveler", "usa" and "visa". But the word "to" would be ignore in this type of search, since it's too generic. In fact, the word "to" changes everything in this, since it refers to where the person wants to travel to (from Brazil to the US). This results in very different outcomes of the search query.

But there are more challenges to this then word relations.

The above shows english language examples. Non-english languages pose a challenge in terms of Machine Learning challenges, since there's not as much data to train models when compared to the english language. However, recent breakthroughs allows having good model results, even in multilingual approaches, including Danish. That's where BERT comes in. BERT is "a neural network-based technique for natural language processing (NLP) pre-training called Bidirectional Encoder Representations from Transformers" (https://blog.google/products/search/search-language-understanding-bert/). This model allows for contextual representation of words. In other words, context matters. Remember the travelling example from above. 

The above techniques set the foundations for this project, since they are part of a toolset to make search engine results better using document similarity. Ultimately, 

> the goal of this project is to determine what role does AI and Machine Learning bring to document searchs, how it combines with existing searches (instead of replacing it) and how it performs in non-English languages.




## Methodology
---

### The dataset

The dataset is comprised of 4.2m jobs from jobindex.dk, from 2000 until March 2020. Here's an excerpt of the dataset:

|    | title                                                                            | location   | description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | company               | date       |
|---:|:---------------------------------------------------------------------------------|:-----------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------|:-----------|
|  0 | Vagtgående maskinmester                                                          | Randers    | Verdo A/S, Randers Verdo ejer og driver kraftvarmeværket i Randers samt varetager drift og overvågning af andre kraftvarmeværker og diverse tilsynsopgaver. Til et af vores 7 vagthold søges snarest muligt en maskinmester. Din primære arbejdsopgave vil være drift, planlægning og overvågning af anlæg på kraftvarmeværket. Herudover udføres der fra kontrolrummet drift og fjernovervågning af andre kraftværker samt tilsyns- og supportopgaver. Du vil således opleve en varieret arbejdsdag, der kan være alt fra en rolig vagt til kaotisk kriseredning.  Verdo søger for en grundig sidemandsoplæring i driften af vores anlæg, inden du slippes løs på egen hånd.                                                                                                                                                    | Verdo A/S             | 2020-03-09 |
|  1 | Administrativ medarbejder/sekretær til stabsfunktion                             | nan        | Halsnæs KommuneHar du lyst til at stå for de administrative opgaver i et område, der er i rivende udvikling og har stor politisk bevågenhed? Er du samtidig en dygtig, samvittighedsfuld og udviklingsorienteret administrativ medarbejder der kan holde styr på alt fra de administrative processer til ledelsens kalendere? Så har vi jobbet til dig her i Halsnæs Kommune. Vi søger en administrativ medarbejder til stabsfunktionen i Område for Ejendomme. Området står midt i en stor omstillingsproces og udvider netop nu med flere opgaver og kompetencer – og denne stilling er netop tænkt til at kunne støtte op om og forankre vores mange nye processer samt sikre, at alle de administrative opgaver løses, så vores projektledere, driftskonsulenter og andre specialister kan koncentrere sig om deres opgaver. | Halsnæs Kommune       | 2020-03-09 |
|  2 | Social- og sundhedsassistent i fortrinsvis dagvagt til Drachmannsvænget i Skagen | nan        | Frederikshavn KommuneØnsker du et varieret og spændende job inden for demensafsnittet, hvor alle bidrager aktivt i opgaveløsningen? Og har du lyst til at arbejde primært i fast dagvagt og have weekendvagt i ulige uger?  Vi søger en medarbejder med sundhedsfaglig baggrund til borgere med demens. Stillingen er på gennemsnitlig 30 timer pr. uge. Du kommer både selvstændigt og i samarbejde med andre samarbejdspartnere til, at arbejde med udgangspunkt i borgerens behov og værdier. Dette sker ud fra den rehabiliterende tilgang hvor du er med til at bevare/forbedre borgernes fysiske, psykiske og sociale funktioner. Det er således vigtigt, at det faldet dig naturligt at være fleksibel og du er nærværende og tillidsskabende i relationen.                                                               | Frederikshavn Kommune | 2020-03-09 |
|  3 | Test Manager                                                                     | København  | LB Forsikring, København  Vi tilbyder opgaver og projekter, der optimerer og digitaliserer vores forretning, og her har du mulighed for at tage ansvar og bruge din faglighed og energi til at føre os ind i en fremtid, hvor vores medlemmer stiller andre og markant større krav til os som forsikringsfællesskab.  I rollen som test manager får du ansvaret for at definere, hvordan testen skal kvalitetssikre de systemer og applikationer, vi arbejder på. Du kommer ligeledes til at supportere i udarbejdelse af QA-strategi og -procedure i LB. At varetage planlægning og koordinering af QA-aktiviteter på tværs af projekter og afdelinger er også et vigtigt formål i rollen.                                                                                                                                      | LB Forsikring         | 2020-03-09 |
|  4 | Lærere til mellemtrinnet på Møn skole - afdelingen i Stege                       | nan        | Vordingborg KommuneHar du tæft for specialpædagogisk virke og inkluderende læringsmiljøer, og vil du indgå i forpligtende kollegiale læringsfællesskaber? Så er Møn skole stedet for dig.  På Møn skole har vi pr. 1/5-20 en fast stilling og en barselsstilling ledig på mellemtrinnet på afdelingen i Stege.  Vi søger to fagligt dygtige lærere, der er kompetente til at undervise i flere af fagene dansk, engelsk, kristendom, musik, håndværk og design og som kan undervise på fleksible hold med elever med behov for specialpædagogisk bistand.  Det forventes, at du som ansøger har it- og iPad-kundskaber på brugerniveau. Det forventes yderligere, at du er indstillet på at arbejde med bevægelse, CL-strukturer og co-teaching i undervisningen.                                                                | Vordingborg Kommune   | 2020-03-09 |


_Title_, _location_, _description_ and _date_ are some of the features in the original dataset. The remainder were excluded since they don't provide signal to the project at hand. Some of the entries have empty rows for certain features. This is due to some difficulty upon scraping the data.

### The hardware

The computations were made in a Macbook Pro 2016 model.

### Preprocessing

The preprocessing of text task included several subtasks:

1. Lower casing of words
2. Lemmatization
3. Stemming
4. Removing stop words in Danish

Apart from lemmatization and stemming - already explained in the Introduction - it's necessary to remove stop words. Stop words are common words on a given language. Some of the danish stop words are: 

```
'og',
 'i',
 'jeg',
 'det',
 'at',
 'en',
 'den',
```

And the list goes on.

Since some job titles and descriptions were written in danish and others where written in english, it was a challenge to differentiate each, so that the preprocessing occured separately for both languages. There are a few possiblities here to disntiguish languages, such as using a language detector, but it is something left for a future iteration of this project.

### Techniques and models used

#### Direct search

Direct search is a simple search based on query terms. As mentioned above, if you search for the word "teacher", you'll get the results regarding job titles that have the same word included. There are no variances to this. 

#### Term Frequency–Inverse Document Frequency (TFIDF) 

This numerical statistic determines the amount of times a words appears in a document, balanced by the inverse number of documents where that word appears. A word that appears a lot in one document would be balanced by how many document it appears. Simply put, considering TFIDF returns a value, if you have a word that appears once in many different documents and another that appears many times in the same document, they would have very similar values.

In this project, this is then calculated for every single word of the dataset. Of course, each word will have its own vectorial representation, the denominated "word embeddings" ([further reading](https://en.wikipedia.org/wiki/Word_embedding)).

TFIDF is a technique that assess a word without its context.

#### Bidirectional Encoder Representations from Transformers (BERT)

BERT is a pre-trained open-source language model, as an outcome from a [paper](https://arxiv.org/abs/1810.04805v2) published in 2018 by the researchers at Google. BERT introduces a contextual bidirectional approach. Simply put, it looks at a setence and its context by "reading" it from left to right and right to left. [Further reading here](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html).

In this project, and due to computational limitations, BERT was not trained again using Jobindex's data. 

### Model assessment
By creating the vectorial representations of job postings it's then possible to compare how similar then are with one another. This similarity is determined via 'cosine similarity' ([Further reading here](https://en.wikipedia.org/wiki/Cosine_similarity)) and all the distances are stored in a distance matrix. 

## Results
---

The results results from the actual jobindex.dk website are then compared with direct search, TFIDF and BERT.

The direct search is a very simple approach to return direct results, as explained before. Without going into details, it's limited into one function:

```
def find_direct_results(search_query):
    matching_entries = [df['title_processed'].index[df['title_processed'].str.contains(word, case=False)]
                        .values for word in search_query.split()]
    return list(set(matching_entries[0]).intersection(*matching_entries))
```


Obviously, today there are very advanced techniques and libraries that work very well. As mentioned before, the goal is to determine on can AI leverage existing search techniques.

However, it was not possible to perform computations on the whole dataset due to its size. This is mainly because when storing the distances from TFIDF, it creates a distance matrix of 4.2m rows x 4.2m colums (resulting in 1.764e+13 cells), making it impossible to store in disk on the current machine used and virtually impossible to store in memory on any machine. A possiblity to tackle this challenge is presented in the Future Improvements chapter later on. 

The short-term solution was to create slice of the original dataset. Thus, a dataset comprised of all the datasets between 2020-02-26 and 2020-03-26 (inc.) was created. This results in dataset with the size of 10140 job postings.

BERT however, was handled in a slightly different way than the TFIDF approach. A file with all the embeddings (of the above subset of data) is created and later on imported in memory. The reason why the last step before getting recommendations from BERT is handled in memory is because of its internal optimizations (since it uses tensors, instead of regular arrays). Being able to quickly load and sort a dataset of around 10k rows is one of the advantages from BERT when compared to TFIDF. Of course, for bigger datasets, a better form of handling would be to have disk persistance. 

The [github repository](GITHUB REPO) shows the preprocessing in detail.

In order to compare results, three types of examples of job searches were used. A very short query term is the first example, followed by a medium (2 words) and finished by a long query (3+ words), as show below

### Example 1 - "læge neuropædiatri": 

This search query roughly translates to "doctor neuropediatry".

In this example, all models are compared, including also the results from jobindex. So, for this query, the results are:





### Example 2


### Example 3



## Caveats
---




## Limitiations and Future improvements
- Improve direct search
- Train BERT to improve results
- Don't load data in memory
- Improve model performance assessment
- Distinguish English from Danish
- Descriptions are not representative

Conclusion

- Use AI on top of traditional search
- TFIDF succeeds in short queries
- BERT succeeeds in 2 or more word queries
