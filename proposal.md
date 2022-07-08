# Challenge Introduction
Learning to Rank~(LTR), aiming to measure documents' relevance w.r.t. queries, is a popular research topic in information retrieval with huge practical usage in web search engines, e-commerce, and multiple different streaming services. With the vogue of deep learning, the heavy burden of data annotation drives the academia and industry communities to the study of learning to rank using implicit user feedback or pre-training language model~(PLM) with self-supervised learning. 

However, directly optimizing the model with click data results in unsatisfied performance due to the strong bias on implicit user feedback, such as position bias, trust bias, and click necessary bias. Unbiased learning to rank~(ULTR) is then proposed for debiasing user feedback with counterfactual learning algorithms. However, real-world user feedback can be more complex than synthetic feedback generated with specific user behavior assumptions like position-dependent click model and ULTR algorithms with good performance on synthetic datasets may not show consistent good performance on the real-world scenario.

Furthermore, it is nontrivial to directly apply the recent advancements in PLMs to web-scale search engine systems since explicitly capturing the comprehensive relevance between query and documents is crucial to the ranking task. However, existing pre-training objectives, either sequence-based tasks (e.g., masked token prediction) or sentence pair-based tasks (e.g., permuted language modeling), learn contextual representations based on the intra/inter-sentence coherence relationship, which cannot be straightforwardly adapted to model the query-document relevance relations.
Although user behavioral information can be leveraged to mitigate this defect, elaborately designing relevance-oriented pre-training strategies to fully exploit the power of PLMs for industrial ranking remains elusive, especially in noisy clicks and exposure bias induced by the search engine.

Baidu is the bigest Chinese search engine, with nearly 8 billion hours spent on Baidu app each month and over 500 million monthly active users. It has a great ambition to promote the techniques development in related topics. Therefore, we host these two challenges for the development of whole community.


# Task Description & Data
- Metric

  - The following evaluation metric is employed to assess the performance of the ranking system. The Discounted Cumulative Gain (DCG) is a standard listwise accuracy metric and is widely adopted in the context of ad-hoc retrieval. For a ranked list of N documents, we use the following implementation of DCG: 
   $DCG@N = \sum_{i=1}^N \frac{G_i}{\log_2(i+1)}$
   where $G_i$ represents the weight assigned to the document’s label at position $i$. A higher degree of relevance corresponds to a higher weight. We use the symbol DCG to indicate the average value of this metric over the test queries. DCG will be reported only when absolute relevance judgments are available.

- Unbiased Learning to Rank
 
 For unbiased learning to rank, 
 

- Pre-training for Web Search
 
 For pre-training, 

- Dataset 
 - The dataset is aviable at .

# Tool


# Timeline

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
