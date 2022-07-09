# Challenge Introduction
Learning to Rank~(LTR), aiming to measure documents' relevance w.r.t. queries, is a popular research topic in information retrieval with huge practical usage in web search engines, e-commerce, and multiple different streaming services. With the vogue of deep learning, the heavy burden of data annotation drives the academia and industry communities to the study of learning to rank using implicit user feedback or pre-training language model~(PLM) with self-supervised learning. However, directly optimizing the model with click data results in unsatisfied performance due to the strong bias on implicit user feedback, such as position bias, trust bias, and click necessary bias. Unbiased learning to rank~(ULTR) is then proposed for debiasing user feedback with counterfactual learning algorithms. However, real-world user feedback can be more complex than synthetic feedback generated with specific user behavior assumptions like position-dependent click model and ULTR algorithms with good performance on synthetic datasets may not show consistently good performance in the real-world scenario. Furthermore, it is nontrivial to directly apply the recent advancements in PLMs to web-scale search engine systems since explicitly capturing the comprehensive relevance between queries and documents is crucial to the ranking task. However, existing pre-training objectives, either sequence-based tasks (e.g., masked token prediction) or sentence pair-based tasks (e.g., permuted language modeling), learn contextual representations based on the intra/inter-sentence coherence relationship, which cannot be straightforwardly adapted to model the query-document relevance relations. Therefore, in this competition, we focus on unbiased learning to rank and pre-training for web search under real long-tail user feedback dataset from Baidu Search (Baidu is the biggest Chinese search engine with 6.32 million monthly active users that has a great ambition and responsibility to promote the technique development in the community). 

# Dataset & Task Description
## [Dataset](https://arxiv.org/pdf/2207.03051.pdf)
  - Training Dataset: [Large Scale Web Search Session Data](https://drive.google.com/drive/folders/1Q3bzSgiGh1D5iunRky6mb89LpxfAO73J?usp=sharing)
  
  - Validation Dataset: [Expert Annotation Dataset](https://drive.google.com/file/d/1hdWRRSMrCnQxilYfjTx8RhW3XTgiSd9Q/view?usp=sharing) 

### Train Data --- Large Scale Web Search Session Data
The large scale web search session are available at [here](https://drive.google.com/drive/folders/1Q3bzSgiGh1D5iunRky6mb89LpxfAO73J?usp=sharing).
The search session is organized as:
```
Qid, Query, Query Reformulation
Pos 1, URL MD5, Title, Abstract, Multimedia Type, Click, -, -, Skip, SERP Height, Displayed Time, Displayed Time Middle, First Click, Displayed Count, SERP's Max Show Height, Slipoff Count After Click, Dwelling Time , Displayed Time Top, SERP to Top , Displayed Count Top, Displayed Count Bottom, Slipoff Count, -, Final Click, Displayed Time Bottom, Click Count, Displayed Count, -, Last Click , Reverse Display Count, Displayed Count Middle, -
Pos 2, URL MD5, Title, Abstract, Multimedia Type, Click, -, -, Skip, SERP Height, Displayed Time, Displayed Time Middle, First Click, Displayed Count, SERP's Max Show Height, Slipoff Count After Click, Dwelling Time , Displayed Time Top, SERP to Top , Displayed Count Top, Displayed Count Bottom, Slipoff Count, -, Final Click, Displayed Time Bottom, Click Count, Displayed Count, -, Last Click , Reverse Display Count, Displayed Count Middle, -
......
Pos N, URL MD5, Title, Abstract, Multimedia Type, Click, -, -, Skip, SERP Height, Displayed Time, Displayed Time Middle, First Click, Displayed Count, SERP's Max Show Height, Slipoff Count After Click, Dwelling Time , Displayed Time Top, SERP to Top , Displayed Count Top, Displayed Count Bottom, Slipoff Count, -, Final Click, Displayed Time Bottom, Click Count, Displayed Count, -, Last Click , Reverse Display Count, Displayed Count Middle, -
# SERP is the abbreviation of search result page.
```


|Column Id|Explaination|Remark|
|:---|:---|:---|
|Qid|query id||
|Query|The user issued query|Sequential token ids separated by "\x01".|
|Query Reformulation|The subsequent queries issued by users under the same search goal.|Sequential token ids separated by "\x01".|
|Pos|The document’s displaying order on the screen.|\[1,30\]|
|Url_md5|The md5 for identifying the url||
|Title|The title of document.|Sequential token ids separated by "\x01".|
|Abstract|A query-related brief introduction of the document under the title.|Sequential token ids separated by "\x01".|
|Multimedia Type|The type of url, for example, advertisement, videos, maps.|int|
|Click|Whether the user clicked the document.|\[0,1\]|
|-|-|-|
|-|-|-|
|Skip|Whether the user skipped the document on the screen.|\[0,1\]|
|SERP Height|The vertical pixels of SERP on the screen.|Continuous Value|
|Displayed Time|The document's display time on the screen.|Continuous Value|
|Displayed Time Middle|The document’s display time on the middle 1/3 of the screen.|Continuous Value|
|First Click|The identifier of users’ first click in a query.|\[0,1\]|
|Displayed Count|The document’s display count on the screen.|Discrete Number|
|SERP's Max Show Height|The max vertical pixels of SERP on the screen.|Continuous Value|
|Slipoff Count After Click |The count of slipoff after user click the document.|Discrete Number|
|Dwelling Time|The length of time a user spends looking at a document after they’ve clicked a link on a SERP page, but before clicking back to the SERP results.|Continuous Value|
|Displayed Time Top|The document’s display time on the top 1/3 of screen.|Continuous Value|
|SERP to Top|The vertical pixels of the SERP to the top of the screen.|Continuous Value|
|Displayed Count Top|The document’s display count on the top 1/3 of screen.|Discrete Number|
|Displayed Count Bottom|The document’s display count on the bottom 1/3 of screen.|Discrete Number|
|Slipoff Count|The count of document being slipped off the screen.||
|-|-|-|
|Final Click |The identifier of users’ last click in a query session.||
|Displayed Time Bottom|The document’s display time on the bottom 1/3 of screen.|Continuous Value|
|Click Count|The document’s click count.|Discrete Number|
|Displayed Count|The document’s display count on the screen.|Discrete Number|
|-|-|-|
|Last Click |The identifier of users’ last click in a query.|Discrete Number|
|Reverse Display Count|The document’s display count of user view with a reverse browse order from bottom to the top.|Discrete Number|
|Displayed Count Middle|The document’s display count on the middle 1/3 of screen.|Discrete Number|
|-|-|-|

### Validation Dataset --- Expert Annotation Dataset 
The expert annotation dataset is aviable at [here](https://drive.google.com/drive/folders/1AmLTDNVltS02cBMIVJJLfVc_xIrLA2cL?usp=sharing).
The Schema of the [nips_annotation_data_0522.txt](https://drive.google.com/file/d/1hdWRRSMrCnQxilYfjTx8RhW3XTgiSd9Q/view?usp=sharing):
|Columns|Explaination|Remark|
|:---|:---|:---|
|Query|The user issued query|Sequential token ids separated by "\x01".|
|Title|The title of document.|Sequential token ids separated by "\x01".|
|Abstract|A query-related brief introduction of the document under the title.|Sequential token ids separated by "\x01".|
|Label|Expert annotation label.|\[0,4\]|
|Bucket|The queries are descendingly split into 10 buckets according to their monthly search frequency, i.e., bucket 0, bucket 1, and bucket 2 are high-frequency queries while bucket 7, bucket 8, and bucket 9 are the tail queries|\[0,9\]|

The [unigram_dict_0510_tokens.txt](https://drive.google.com/file/d/1HZ7l7UDMH9WvLVoDu-_uqLNjF5gtBe2g/view?usp=sharing) is a unigram set that records the high-frequency words using the desensitization token id.

## Unbiased Learning to Rank
  For the unbiased learning to rank task, you are required to train a ranking model with the Large Scale Web Search Session Data. However, **the Expert Annotation Dataset and extra datasets are not allowed for training the ranking model**. 

## Pre-training for Web Search
 For pre-training for web search task, you are required to pre-train a PLM with the Large Scale Web Search Session Data and finetune the PLM with **the Expert Annotation Dataset**~(Here is the [PLM](https://github.com/ChuXiaokai/baidu_ultr_dataset) for reference).

## Metric
  The following evaluation metric is employed to assess the performance of the ranking system. The Discounted Cumulative Gain (DCG) is a standard listwise accuracy metric and is widely adopted in the context of ad-hoc retrieval. For a ranked list of N documents, we use the following implementation of DCG: 
   $DCG@N = \sum_{i=1}^N \frac{G_i}{\log_2(i+1)}$
   where $G_i$ represents the weight assigned to the document’s label at position $i$. A higher degree of relevance corresponds to a higher weight. We use the symbol DCG to indicate the average value of this metric over the test queries. DCG will be reported only when absolute relevance judgments are available. Particularly, we employ the DCG@4 as the final metrics since the top-4 results are firstly presented to users.

# Timeline

|Event|Complete Date|
|:---|:---|
|Oct 15, 2022|	Website ready and training set available for download.|
|Nov 11 2022|	Intermediate test set release and intermediate submission starts.|
|Dec 11 2022| Dec 20 2021	Intermediate submission ends.|
|Dec 16 2022| Dec 22 2021	Intermediate leaderboard result announcement.|
|Dec 17 2022| Dec 23 2021	Final test set release and final submission starts.|
|Jan 20 2023|	Final submission ends.|
|Jan 24 2023|	Final leaderboard result announcement.|
|Jan 25 2023| Invitations to top 6 teams for short papers.|
|Feb 15 2023|	Short paper deadline.|
|Feb 21-25 2023|	WSDM Cup conference presentation.|



# Prizes:  
- Unbiased Learning to Rank
  
  - Champion: One team ($2000) 

  - Runner-up: One team ($1000) 

  - 3rd-place: One team ($500)
  
- Pre-training for Web Search 

  - Champion: One team ($2000) 

  - Runner-up: One team ($1000) 

  - 3rd-place: One team ($500)

# Grand Challenge Contacts

- Lixin Zou (zoulixin15@gmail.com)
- Haitao Mao 
- Xiaochai Chu
- Changying Hao
- Shuaiqiang Wang
- Dawei Yin
