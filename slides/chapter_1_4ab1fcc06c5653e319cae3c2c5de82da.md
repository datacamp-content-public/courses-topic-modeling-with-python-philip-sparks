---
title: Insert title here
key: 4ab1fcc06c5653e319cae3c2c5de82da

---
## Latent Semantic Analysis

```yaml
type: "TitleSlide"
key: "b29b500e97"
```

`@lower_third`

name: Philip Sparks
title: Data Scientist


`@script`
Welcome. We’ve reviewed and cleaned our data. We are now ready to review our first topic modeling algorithm!


---
## What is Latent Semantic Analysis?

```yaml
type: "FullSlide"
key: "625adad5f5"
```

`@part1`
Latent Semantic Analysis (LSA), or sometimes known as Latent Semantic Indexing (LSI), is a topic model technique that reveals words associated with one another by transforming the words into vectors and reducing those vectors into categories.


`@script`
Latent Semantic Analysis (LSA), or sometimes known as Latent Semantic Indexing (LSI), is a topic model technique that reveals words associated with one another by transforming the words into vectors and reducing those vectors into categories.


---
## Intuition

```yaml
type: "FullSlide"
key: "9c38107469"
```

`@part1`
Let’s say you have the following two sentences:

1. There’s a new book you must check out at the library.

2. I need to book a new flight to visit New York.


`@script`
Before we get into the mathematical concept, let’s talk through an example to gain an intuition for what problem LSA solves.

Let’s say you have the following two sentences:

There’s a new book you must check out at the library.
I need to book a new flight to visit New York.

In the first sentence, the word ‘book’ is a noun referring to a physical object. In the second sentence, ‘book’ is an adverb used to signal that someone needs to make a reservation. These are subtleties that are easy for humans to understand, but much more difficult for a computer to detect. Our task is to find the real meaning, and not simply pattern matching, of words inside documents.


---
## Three Step Process for LSA

```yaml
type: "FullSlide"
key: "63b98cd684"
```

`@part1`
1. Constructing a weighted term-document matrix (TF-IDF).
2. Performing a Singular Value Decomposition (SVD) on the matrix, and
3. Using that matrix to identify the concepts or topics contained in the text.


`@script`
That’s when using LSA becomes a valid use case. LSA uses common linear algebra techniques to learn the conceptual correlations in a collection of text. In general, the process involves three simple steps.
1. Constructing a weighted term-document matrix
2. Performing a Singular Value Decomposition on the matrix, and
3. Using that matrix to identify the concepts or topics contained in the text.


---
## Singular Value Decomposition (SVD)

```yaml
type: "FullSlide"
key: "7cf89ef104"
center_content: true
```

`@part1`
![Image](https://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1538411402/image3_maagmh.png)


`@script`
Assume you have m number of documents with n number of unique words. We want to find k topics from all of the text contained in each document. It is up to us to specify the number of  k topics we wish to find. This can be done due to theory in linear algebra that there exists a decomposition of X (our m by m term document matrix) such that our Word Assignment to Topics (m by n) matrix and Topic Distribution Across Documents (n * m) matrix are orthogonal (meaning its transpose is equal to its inverse), while the Topic Importance (n by n) matrix is a diagonal one. This formula is the singular value decomposition.


---
## Weighted-Term Document Index

```yaml
type: "FullCodeSlide"
key: "eb7b54d69a"
center_content: true
```

`@part1`
```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', 
max_features= 1000, # keep top 1000 terms 
max_df = 0.5, 
smooth_idf=True)

X = vectorizer.fit_transform(news_df['clean_doc'])```


`@script`
First, we create a document-term matrix grid of m documents by n words with TF-IDF scores using the TfidfVectorizer from Sci-Kit Learn. We’ll want to set max_features at one-thousand to have enough distinct words for our matrix, but still computationally manageable for our local machine.


---
## TruncatedSVD()

```yaml
type: "FullCodeSlide"
key: "13bfa429d9"
center_content: true
```

`@part1`
```python
from sklearn.decomposition import TruncatedSVD

# SVD represent documents and terms in vectors 
svd_model = TruncatedSVD(n_components=20, algorithm='randomized', n_iter=100, random_state=122)

svd_model.fit(X)
```


`@script`
Then, we will reduce the number of dimensions our matrix has to the k number of topics we want to find using TruncatedSVD. In this instance, we select twenty topics, knowing that our Newsgroups dataset has twenty groups. The number of topics can be specified by using the n_components parameter. LSA can typically uncover 100 or more distinct topics from large corpuses of data.


---
## View Topics

```yaml
type: "FullCodeSlide"
key: "3da632bff4"
```

`@part1`
```python
terms = vectorizer.get_feature_names()

for i, comp in enumerate(svd_model.components_):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:5]
    print("Topic "+str(i)+": ")
    for t in sorted_terms:
        print(t[0])
        print(" ")
```
```python
Topic 0: like know people think good 
Topic 1: thanks windows card drive mail 
Topic 2: game team year games season
... 
Topic 17: drive scsi disk hard card
Topic 18: windows file window files program 
Topic 19: government chip mail space information

```


`@script`
Lastly in this code snippet, we want to enumerate the components in our model to reveal which words comprised each topic. We see the words that comprise each topic and can infer the content of the documents contained in our corpus.


---
## Let's begin!

```yaml
type: "FinalSlide"
key: "f233a4d967"
```

`@script`
Now that we’ve tried latent semantic analysis in this example, let's have you have a go at it!

