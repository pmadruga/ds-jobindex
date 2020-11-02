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

However, recent advances in Natural Language Processing (NLP) techniques have allowed for transforming words into a common base form. For example, words like "am", "are", "is" have their common form is the word "be" (a technique known as "Stemming"). They can also convert the plural of a word to it's singular (known as "Lemmatization"). Here's an example with both techniques: "The boys are eating pizza", after being processed with the above techniques becomes "The boy is eat pizza". Applying these two techniques to documents will definitely improve search results, finding similar results to our original search query. And those techniques are part of a few others used in this project, for a mostly Danish language dataset. This project benefits from how

>  Danish NLP techniques have reached a very good level, 

when compared to English language-based ones (typically more complete).

The relations between words are also important. Let's take for example (based on "https://blog.google/products/search/search-language-understanding-bert/") that you're searching for "2019 brazil traveler to usa need a visa.". You'd probably get a result that includes the words "2019", "brazil", "traveler", "usa" and "visa". But the word "to" would be ignore in this type of search, since it's too generic. In fact, the word "to" changes everything in this, since it refers to where the person wants to travel to (from Brazil to the US). This results in very different outcomes of the search query.

But there are more challenges to this then word relations.

The above shows english language examples. Non-english languages pose a challenge in terms of Machine Learning challenges, since there's not as much data to train models when compared to the english language. However, recent breakthroughs allows having good model results, even in multilingual approaches, including Danish. That's where BERT comes in. BERT is "a neural network-based technique for natural language processing (NLP) pre-training called Bidirectional Encoder Representations from Transformers" (https://blog.google/products/search/search-language-understanding-bert/). This model allows for contextual representation of words. In other words, context matters. Remember the travelling example from above.

> The above techniques set the foundations for this project, since they are part of a toolset to make search engine results better using document similarity.


## Methodology
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

### 
