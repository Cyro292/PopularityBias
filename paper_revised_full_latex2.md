\documentclass[11pt,a4paper,twocolumn]{article}

\usepackage[a4paper,margin=1.8cm]{geometry}
\usepackage{microtype}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage[hidelinks]{hyperref}
\usepackage{url}
\usepackage{graphicx}

\title{Knowledge-Aware Retrieval Systems (KARS)}
\author{Amon Sisowath}
\date{February 2026}

\begin{document}

\maketitle

\begin{abstract}
Later
\end{abstract}
\section{Introduction}

Large Language Models (LLMs) have demonstrated remarkable performance across a wide variety of knowledge-intensive tasks that require the memorization, retrieval, and application of factual knowledge \cite{chowdhery2022palm,yu2022}. However, important limitations remain. LLMs can struggle with less frequent and long-tail facts, generate unsupported statements when factual knowledge is missing or uncertain, and become outdated when the knowledge encoded in their parameters no longer reflects the current state of the world. These limitations are particularly problematic for factual question answering and other knowledge-intensive applications, where correctness depends on access to accurate and up-to-date information.

To address these shortcomings, Retrieval-Augmented Generation (RAG) has emerged as a widely adopted paradigm. Instead of relying solely on parametric memory, RAG systems retrieve relevant information from external knowledge sources and provide this information as context to a language model. This allows generated responses to be grounded in external evidence and can mitigate failures caused by outdated, incomplete, or hallucinated parametric knowledge. Retrieval is also useful for addressing challenges associated with long contexts, such as the well-known ``lost-in-the-middle’’ phenomenon \cite{liu2024lost}, by selectively surfacing relevant information rather than requiring the model to process an entire knowledge source at once.

Underlying many of these advances is embedding-based similarity search, in which learned vector representations are used to identify semantically related information. While often discussed in the context of RAG, its importance extends far beyond retrieval-augmented generation. Embedding-based similarity search also forms the foundation of entity matching and entity linking systems, where it is used to identify corresponding entities across heterogeneous data sources. These capabilities are essential for a wide range of downstream applications, including knowledge graph construction, semantic search, and advanced retrieval architectures such as GraphRAG. Consequently, embedding-based similarity search has become a fundamental building block of modern AI systems. Understanding its behavior is therefore important not only for improving RAG systems, but also for advancing a broader class of knowledge-intensive applications that rely on accurately identifying and connecting relevant information.

While retrieval is often introduced as a corrective mechanism for weaknesses in parametric memory, it is not obvious that retrieval systems perform uniformly across all types of knowledge. In particular, if language models struggle more with long-tail facts, an important question is whether retrieval systems can reliably compensate for this weakness or whether they exhibit their own popularity-related performance differences. This question is especially relevant because popular entities are often better represented, more frequently mentioned, and more richly connected in large knowledge corpora, while less popular entities may be harder to identify, distinguish, or rank correctly.

This thesis therefore investigates whether popularity influences retrieval effectiveness beyond differences in corpus coverage. Our central objective is to determine whether neural retrieval models exhibit popularity-dependent behavior when information availability is held constant and, if so, to identify the mechanisms responsible for such effects. Understanding these factors is important for determining whether popularity-related retrieval disparities originate primarily from the structure of the underlying knowledge corpus or from the retrieval models themselves.

More specifically, this thesis addresses the following research questions:

\begin{enumerate}
\item \textbf{RQ1:} To what extent does retrieval performance differ between popular and unpopular facts?

\item \textbf{RQ2:} When information availability is controlled for, do retrieval models exhibit a systematic tendency to favor popular information during retrieval and ranking?
\item \textbf{RQ3:} Which retrieval characteristics contribute to popularity-related performance differences, and how do they influence retrieval effectiveness?

\end{enumerate}

Motivated by prior findings on popularity effects in factual memorization, we hypothesize that popularity may affect not only the memorization capabilities of language models but also the representations learned by neural retrieval systems. Specifically, we propose that popular entities and facts may be encoded as more distinctive and semantically robust vector representations, making relevant documents easier to identify and rank correctly. Under this hypothesis, the same factors that facilitate the memorization of popular knowledge in language models may also improve retrieval effectiveness, potentially leading to popularity-related performance differences even when information availability is controlled.

To investigate these questions, we evaluate three dense retrieval models based on multilingual sentence embeddings: bilingual-embedding-small \cite{conneau2019unsupervised,reimers2019sentence,thakur2020augmented}, multilingual-e5-small \cite{wang2024multilingual}, and multilingual-e5-large \cite{wang2024multilingual}. Retrieval is performed over the 2019 KILT Wikipedia snapshot \cite{fb_kilt} using FAISS-based indexes. To assess the downstream impact of retrieval quality, retrieved documents are provided as context to two language models, GPT-Neo 2.7B and Qwen 7B, which generate answers to benchmark questions. Experiments are conducted on four knowledge-intensive benchmarks: PopQA, Natural Questions, TriviaQA, and FEVER. All datasets are aligned with the same Wikipedia snapshot to ensure that retrieval results are comparable across benchmarks.

Our main findings can be summarized as follows:

\begin{itemize}
\item Regarding \textbf{RQ1}, we observe substantial retrieval disparities between popular and unpopular facts across all evaluated retrieval systems. Moreover, the performance advantage of dense retrieval over BM25 increases with popularity: while dense retrievers substantially outperform BM25 on highly popular facts, this advantage is considerably smaller for less popular knowledge.

\item Regarding \textbf{RQ2}, we find little evidence that retrieval models systematically retrieve or rank popular documents ahead of equally relevant unpopular documents when document availability is controlled for. This suggests that the popularity-related performance gap observed in realistic retrieval settings is not driven by a direct preference for popular information within the retrieval model itself.
\item Regarding \textbf{RQ3}, our analysis indicates that differences in document distinguishability and retrieval confidence are key factors underlying the popularity-related retrieval gap. Popular facts tend to be associated with more separable representations, making relevant documents easier to identify and rank correctly.

\end{itemize}

Overall, this thesis contributes to a more precise understanding of popularity effects in retrieval systems. Our findings show that retrieval effectiveness varies substantially across the popularity spectrum, yet provide little evidence that retrieval models inherently favor popular documents when information availability is controlled. These results suggest that popularity-related retrieval disparities are more closely associated with the properties of the retrieval task itself than with a direct preference for popular information within the retrieval model.

\section{Related Work}

\paragraph{Parametric and non-parametric knowledge.}
Large language models can store substantial factual knowledge in their parameters and use this knowledge to perform well on knowledge-intensive tasks. Chowdhery et al. \cite{chowdhery2022palm} show that scaling language models leads to strong performance across a broad range of reasoning and factual tasks, while Yu et al. \cite{yu2022} show that large language models can often generate useful contextual information without explicitly retrieving external documents. These findings suggest that modern language models possess strong parametric memories. However, relying exclusively on parametric knowledge has important limitations. Kasai et al. \cite{kasai2022temporal} show that factual question answering becomes difficult when answers change over time, and Jang et al. \cite{jang2022knowledge} similarly argue that language models need mechanisms for handling continuously evolving knowledge. Shuster et al. \cite{shuster2021retrieval} further show that language models may hallucinate when they are not grounded in external evidence, and that retrieval augmentation can reduce such unsupported generations. Together, these works motivate the distinction between parametric knowledge stored in model weights and non-parametric knowledge retrieved from an external corpus.

\paragraph{Popularity and factual memorization.}
A second line of work shows that parametric knowledge is not acquired uniformly. Kandpal et al. \cite{kandpal2022large} investigate long-tail factual knowledge and find that language models are substantially worse at answering questions about less frequent entities. They show that factual recall is strongly related to the amount of relevant evidence observed during pretraining, suggesting that memorization depends heavily on frequency and exposure. Mallen et al. \cite{mallen2023trust} study this relationship through the lens of entity popularity and show that popularity is highly predictive of whether language models can answer factual questions from parametric memory. They find that models are more reliable on popular entities, while less popular entities are more likely to require retrieval. These results establish popularity as an important factor in understanding when parametric knowledge succeeds or fails. However, both Kandpal et al. and Mallen et al. focus primarily on the knowledge stored in language model parameters and on the downstream effect of retrieval augmentation, rather than on whether retrieval systems themselves exhibit popularity-dependent behavior.

\paragraph{Retrieval augmentation for factual knowledge.}
Retrieval-Augmented Generation addresses weaknesses of parametric memory by retrieving external documents and providing them as additional context to the language model. Lewis et al. \cite{lewis2020rag} introduce retrieval-augmented generation for knowledge-intensive NLP tasks and show that combining generation with retrieved evidence improves factual performance. Shuster et al. \cite{shuster2021retrieval} show that retrieval augmentation reduces hallucination in dialogue by grounding responses in retrieved text. Mallen et al. \cite{mallen2023trust} further show that retrieval is especially beneficial for less popular factual knowledge, where parametric memory is less reliable. This makes retrieval a central mechanism for compensating for the long-tail weaknesses of language models. However, if retrieval is used to correct popularity-related failures of parametric memory, then the retrieval component itself must also be examined. In particular, it is not enough to know that retrieval helps less popular facts in downstream QA; we also need to understand whether retrieval models identify and rank relevant documents equally well across the popularity spectrum.

\paragraph{Positioning of this work.}
Prior work has therefore established three important findings. First, large language models can store and use factual knowledge parametrically, but this knowledge can be incomplete, outdated, or hallucinated \cite{shuster2021retrieval,kasai2022temporal,jang2022knowledge}. Second, factual memorization is strongly associated with frequency and popularity: Kandpal et al. \cite{kandpal2022large} show that long-tail knowledge is harder to learn, and Mallen et al. \cite{mallen2023trust} show that entity popularity predicts whether language models can answer factual questions from parametric memory. Third, retrieval augmentation can improve factual answering, particularly when parametric memory is insufficient \cite{lewis2020rag,mallen2023trust}. What remains less clear is whether the retrieval component itself is also affected by popularity. Existing work shows that retrieval can compensate for failures of parametric knowledge, but does not fully explain whether popularity-related retrieval differences arise from corpus availability, from the retrieval model’s representations and ranking behavior, or from the interaction between both. This thesis addresses this gap by studying popularity effects directly in retrieval systems. We analyze how retrieval performance changes across popularity levels and investigate whether dense retrievers systematically favor popular information when information availability is controlled.

\section{Evaluation Setup}

We evaluate popularity effects in an end-to-end open-domain retrieval-augmented generation (RAG) pipeline over a fixed Wikipedia corpus. Given a question, a retriever searches over Wikipedia passage chunks and returns a ranked list of passages. The top-ranked passages are then provided as context to a language model, which generates an answer. We evaluate both retrieval quality and downstream generation accuracy.

\subsection{Task Definition}

We formulate the task as open-domain question answering with retrieval. Given a question, the retriever searches over the full Wikipedia passage index and returns a ranked list of passages. The top-ranked passages are then concatenated and provided as context to a generator, which produces the final answer.

Since questions are linked to target Wikipedia articles, while the retrieval corpus consists of passage chunks, we evaluate retrieval at the article level. A retrieved passage is considered relevant if its metadata points to the same Wikipedia article as the question’s target article. Thus, the retriever is not required to recover a specific annotated span; it only has to retrieve at least one passage from the correct article.

Retrieval performance is measured using Recall@10 and MRR. If no passage from the target article is retrieved, the reciprocal-rank contribution is set to zero.

To evaluate downstream generation, we report answer accuracy using both substring matching and binary LLM judging. Substring accuracy marks a generated response as correct if it contains any accepted gold answer. The LLM judge provides a more flexible binary assessment by evaluating whether the generated response correctly answers the question, even when the wording does not exactly match the reference answer.

\subsection{Dimensions of Analysis}

Popularity is the central dimension of our analysis. Prior work often measures popularity internally, for example through entity frequency or subject-object co-occurrence counts in model training data. Such measures are useful for studying memorization, and Kandpal et al.~\cite{kandpal2022large} show that they strongly predict factual recall.

However, internal popularity is difficult to use in our setting. Training corpora are model-specific, often unavailable, and not directly comparable across retrievers and generators. We therefore use external popularity as our main measure. Following Mallen et al.~\cite{mallen2023trust}, we measure entity popularity using Wikipedia page views. Page views are model-independent, available for the Wikipedia articles in our corpus, and provide a practical proxy for real-world entity visibility.

After defining the popularity measure, we need to specify how popularity groups are constructed. A standard approach is to compute deciles over articles. However, this can be misleading in retrieval because popular articles are often longer and contribute more passage chunks to the index. Since retrieval happens over chunks rather than full articles, popular articles may have more chances to be retrieved simply because they occupy more index space.

We therefore distinguish two mechanisms. The first is exposure bias: popular articles may perform better because they contribute more retrievable passages. The second is inherent retrieval bias: retrievers may still perform differently across popularity groups even when each group contributes a comparable number of chunks.

Our main analysis focuses on the second mechanism. We therefore use chunk-weighted popularity deciles. In this setting, articles are weighted by the number of passages they contribute to the retrieval index when decile boundaries are computed. As a result, each popularity decile contains approximately the same number of retrievable chunks. This makes retrieval comparisons less dependent on unequal index exposure.

\subsection{Benchmarks}

We evaluate on several Wikipedia-linked knowledge-intensive datasets. This reduces dependence on a single benchmark and allows us to test popularity effects across different question types.

PopQA is an entity-centric open-domain QA dataset designed to cover entities with a wide range of popularity \cite{mallen2023trust}. It is especially relevant for this study because popularity variation is central to its construction. Unlike the other datasets used here, PopQA is not sampled from KILT.

Natural Questions contains real user questions issued to Google Search and paired with Wikipedia evidence \cite{petroni2021kilt}. It serves as a standard open-domain QA benchmark. TriviaQA contains question-answer pairs collected from trivia sources and linked to evidence documents \cite{petroni2021kilt}. It adds a different question style and entity distribution from Natural Questions.

FEVER is a fact verification dataset in which claims are verified against Wikipedia evidence \cite{petroni2021kilt}. We use its claims as additional Wikipedia-linked retrieval queries. HotpotQA contains multi-hop questions that often require reasoning over multiple Wikipedia articles \cite{petroni2021kilt}. In our setup, retrieval is evaluated with respect to the linked target articles. T-REx is derived from Wikidata triples aligned with Wikipedia text \cite{petroni2021kilt}. It provides entity-relation queries and complements the natural-language QA datasets.

After linking and filtering examples against the 2019 Wikipedia snapshot, we construct two evaluation pools. To avoid domination by highly represented popularity groups, we sample questions approximately equally across popularity deciles. Table~\ref{tab:question_distribution} shows the number of retained questions from each dataset.

\begin{table}[t]
\centering
\small
\begin{tabular}{|l|r|r|}
\hline
\textbf{Dataset} & \textbf{8k pool} & \textbf{60k pool} \\
\hline
FEVER & 1064 & 5451 \\
HotpotQA & 1325 & 9675 \\
Natural Questions & 1212 & 6403 \\
PopQA & 1330 & 9702 \\
T-REx & 1329 & 9953 \\
TriviaQA & 1123 & 6007 \\
\hline
\textbf{Total} & \textbf{7383} & \textbf{47191} \\
\hline
\end{tabular}
\caption{Question distribution by source dataset after Wikipedia linking and filtering.}
\label{tab:question_distribution}
\end{table}

The smaller pool is used for repeated retrieval runs and diagnostic analyses. The larger pool is used to estimate retrieval performance over a broader set of questions.

\subsection{Retrievers}

We compare sparse, dense and hybrid retrieval systems. Sparse retrieval tests lexical matching, dense retrieval tests semantic embedding-based retrieval, and hybrid systems test whether combining both signals reduces popularity-related disparities.

For sparse retrieval, we use the local \texttt{bm25s} backend \cite{bm25s}. The main sparse variant is \texttt{bm25\_plus}, with k1 = 1.5, b = 0.75, and delta = 1.0.

For dense retrieval, we evaluate multilingual sentence-embedding models from the MTEB ecosystem:
\begin{itemize}
    \item \texttt{Lajavaness/bilingual-embedding-small}, 384 dimensions \cite{conneau2019unsupervised,reimers2019sentence,thakur2020augmented}
    \item \texttt{intfloat/multilingual-e5-small}, 384 dimensions \cite{wang2024multilingual}
    \item \texttt{intfloat/multilingual-e5-large}, 1024 dimensions \cite{wang2024multilingual}
\end{itemize}

Dense indexes are implemented with FAISS. In the large-scale experiments, we use IVF-PQ indexes and vary the search budget through the number of probed inverted lists.

Hybrid retrieval combines sparse and dense retrieval signals use a rrf fusion.

\subsection{Generators}

After retrieval, the top retrieved passages are passed to a language model. In the main experiments, we use the top three retrieved passages. This evaluates whether retrieved evidence from the target article helps the generator answer correctly.

The prompt is kept fixed across retrieval systems:
\begin{quote}
Documents: \{documents\}

Question: \{question\}
\end{quote}

Here, \{documents\} denotes the concatenation of the retrieved passage texts. We use a simple prompt to reduce the influence of prompt engineering. We also include a zero-shot setting without retrieved documents, which measures how often the generator answers from parametric knowledge alone.

We evaluate two generators:
\begin{itemize}
    \item \textbf{GPT-Neo-2.7B}: \texttt{EleutherAI/gpt-neo-2.7B} \cite{black2021gptneo}
    \item \textbf{Qwen2.5-7B-Instruct}: \texttt{Qwen/Qwen2.5-7B-Instruct} \cite{yang2024qwen25}
\end{itemize}

Following Mallen et al.~\cite{mallen2023trust}, we first use case-insensitive substring matching. A generated answer is correct if any accepted gold answer appears in the generated response. Generation accuracy is the mean correctness across questions.

We also use an LLM-based binary evaluator. The judge receives the question, the generated answer, and the gold reference document, and predicts whether the answer correctly answers the question. We include this metric because Arabzadeh et al.~\cite{arabzadeh2025benchmarking} find that binary LLM relevance judgments align well with human preferences.

We report both substring accuracy and binary accuracy overall and by popularity decile. This allows us to test whether generation disparities across popularity groups remain visible under both exact-match-style and judge-based evaluation.

\subsection{Analogue Similarity Diagnostic}

In addition, we test whether dense retrieval scores reflect representation strength rather than lexical overlap alone. For this purpose, we construct an approximate diagnostic set of 50 entity pairs. Each pair contains two structurally similar Wikipedia entities from the same broad category: a newer entity from the 2026 Wikipedia snapshot, which should not be present in the retriever's training data, and an older, more popular analogue. For each entity, we write matched queries that express the same relation whenever possible.

This diagnostic is not a benchmark. It is a controlled probe of representation quality. We compare dense embedding similarity between the query and the target article in the 2026 snapshot, and compare this against BM25 as a lexical control. BM25 is expected to remain similar when the paired queries have comparable surface terms. A larger dense advantage for older or more popular analogues therefore suggests a representation-strength effect rather than only lexical overlap.

\begin{center}
\Large\bfseries
WORK IN PROGRESS

\vspace{1em}

\normalsize
Please do not read beyond this page.
Subsequent sections are incomplete and may contain
unfinished analyses, missing references, or draft text.
\end{center}


\section{Findings}

We evaluate whether retrieval performance changes with target-article popularity and whether the observed differences are associated with coverage, lexical competition, dense representation quality, or a direct preference for popular documents.

\subsection{Retrieval Performance Depends on Popularity}

\paragraph{Popularity widens the gap between dense retrieval and BM25.}
Figure~\ref{fig:per_decile_retrieval} shows Recall@10 across chunk-weighted popularity deciles. The central pattern is a widening gap between dense retrieval and BM25. This gap is not caused by a single uniform popularity effect. Instead, it results from two separate trends: BM25 performance decreases in the higher-popularity deciles, while dense retrieval performance increases or remains comparatively stable as popularity rises.

Figure~\ref{fig:delta_vs_bm25} shows this divergence directly. For low-popularity targets, dense retrieval provides only a limited advantage over BM25. For high-popularity targets, the advantage becomes substantially larger because the two retrieval methods move in opposite directions. The current BM25 diagnostics do not identify a single cause for this decline, whereas dense retrievers appear to benefit from stronger semantic representations of popular entities. This answers RQ1: popularity affects retrieval performance, but the direction and magnitude of the effect depend strongly on the retrieval method.

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{data/recall10_pop.png}
    \caption{Recall@10 by chunk-weighted popularity decile. Dense retrieval remains stable or improves for popular targets, whereas BM25 weakens in the high-popularity region.}
    \label{fig:per_decile_retrieval}
\end{figure}

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{data/delta_vs_bm25_retrieved_docs_ivfpq_high.png}
    \caption{Recall@10 difference between dense retrieval and BM25 by dataset and popularity decile. The dense--BM25 advantage grows toward the high-popularity deciles.}
    \label{fig:delta_vs_bm25}
\end{figure}

\paragraph{BM25 weakens despite increasing target-side lexical evidence.}
BM25 does not benefit monotonically from popularity. This is notable because popular target articles are substantially longer and contribute many more chunks to the index. Under article-level evaluation, any retrieved chunk from the target article counts as correct. Additional target chunks could therefore be expected to increase the probability that BM25 retrieves the correct article.

The observed pattern is the opposite. For BM25+, hit@1 falls from 43.8\% in the lowest chunk-weighted popularity decile to 15.4\% in the highest decile, corresponding to a 64.7\% relative decrease. This decline occurs even though target-side lexical evidence increases: the best-matching target chunks show greater query-term overlap, higher term frequency, and higher TF--IDF evidence, while popular target articles contain substantially more matching chunks. High-popularity targets are therefore not more difficult because their articles lack lexical evidence.

\paragraph{IDF, TF, and competition diagnostics separate evidence from discriminability.}
Figure~\ref{fig:bm25_degen} summarizes three BM25-relevant mechanisms across chunk-weighted popularity deciles: query-term IDF, query-term TF in the best-matching target chunk, and the fraction of target chunks matched by the query. Mean query-term IDF decreases moderately, from approximately 5.15 to 4.76, while target-side overlap, TF, TF--IDF evidence, and matching-chunk availability all increase substantially. These are opposing effects on BM25 scoring: lower IDF reduces query specificity, whereas stronger target-side evidence should improve retrieval.

Direct BM25 scoring identifies the residual mechanism. At depth 100, the target article remains in the candidate set at essentially the same rate in the lowest and highest deciles (80.1\% and 78.6\%, respectively; $p=0.57$). The failure is therefore primarily a ranking, rather than a coverage, failure. Conditional on the target being retrieved, its mean rank deteriorates from 6.13 to 13.95, and the margin between the best target chunk and the best non-target chunk reverses from $+1.32$ to $-1.41$ ($p<0.001$).

This loss of margin is direct evidence of lexical competition. The number of non-target chunks within 5\% of the best target score increases from 19.4 to 49.6 ($p<0.001$). Although the target's best BM25 score decreases from 46.46 to 41.50, the strongest non-target score changes much less, from 45.15 to 42.91. Popular targets are therefore not absent from the ranked candidate set; they are increasingly outranked by lexically similar distractors.

An identifiable contributor to this competition is reduced discriminability of entity-identifying terms. The probability that a query shares at least one token with its target title is almost unchanged across the endpoint deciles (87.7\% versus 87.3\%), but the BM25 IDF mass of the shared query--title terms falls from 13.84 to 9.46 (31.6\%, $p<0.001$). Thus, popular entities are still named in the query, but their identifying terms are more common in the corpus and provide weaker lexical anchors. This makes many non-target chunks plausible matches. Raw target TF and TF--IDF proxies do not offset this effect because BM25 saturates repeated terms and ranks against these competing chunks rather than against the target alone.

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{data/bm25_degen.png}
    \caption{BM25 lexical diagnostics across chunk-weighted popularity deciles. BM25 hit@1 decreases sharply with popularity even though target-side query-term evidence increases and query-term IDF decreases moderately. Direct score analysis attributes the residual decline to weaker entity-term discriminability and more near-tied non-target chunks.}
    \label{fig:bm25_degen}
\end{figure}

\paragraph{Implications.}
The main implication is that greater index exposure does not necessarily translate into better BM25 retrieval. Although popular articles contribute more chunks and therefore contain more opportunities for lexical matches, their entity cues are less discriminative and generate a denser set of near-tied distractors. The observed BM25 decline is therefore a lexical-ranking failure, not a lack of target-side evidence or target-candidate coverage.

\subsection{Dense Retrieval Is Stronger for Popular Knowledge}

\paragraph{Dense scores increase for popular or historically represented entities.}
Figure~\ref{fig:similarity_analysis} shows the analogue similarity diagnostic. We compare approximately 50 manually selected entity pairs with matched query structures. Each pair contains a newer or less represented entity and an older, more popular analogue from the same broad category.

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{data/similarity_analysis.png}
    \caption{Similarity analysis over manually selected analogue pairs. Dense similarity is significantly higher for older and more popular analogues, whereas the corresponding BM25 differences are not significant.}
    \label{fig:similarity_analysis}
\end{figure}

BM25 serves as a lexical control. Differences in BM25 scores between the analogue groups are not statistically significant, indicating that the paired queries contain comparable lexical evidence. Dense similarity, however, is significantly higher for the older and more popular analogues. This supports RQ3: popular or historically represented entities appear to receive stronger or more separable dense representations.

\paragraph{Implications.}
These findings suggest that the popularity bias previously identified by Mallen et al. and Kandpal et al. is not limited to the generation stage, but can emerge during retrieval. This challenges the view of retrieval-based components as a neutral or comparatively safe alternative to parametric generation. Systems such as GraphRAG, entity matching, semantic linking, and other methods that rely on dense similarity may reproduce the same popularity-related disparities. Moreover, because this study examines only popularity, such systems may also inherit other representation biases that are encoded in their training data but are not analyzed here.

\subsection{Popularity Effects Are Not Direct Popularity Preference}

\paragraph{Wrong-document analysis separates performance from preference.}
The previous results show that retrieval performance varies with target popularity. However, this does not by itself imply that retrievers directly prefer popular documents. To test this, we analyze only incorrect retrievals and compare the popularity of the target article with that of the incorrectly retrieved article.

For target decile $d$, we compute
\begin{equation}
    \mathrm{Pref}(d)
    =
    \frac{1}{|E_d|}
    \sum_{(q,r) \in E_d}
    \mathbb{I}\left[\mathrm{pop}(r) > \mathrm{pop}(a_q)\right],
\end{equation}
where $E_d$ is the set of incorrect retrievals for queries whose target article $a_q$ belongs to decile $d$, and $r$ is an incorrectly retrieved article.

Figure~\ref{fig:wrong_doc_pref} shows this preference under chunk-weighted deciles. The interpretation depends strongly on the chosen baseline. Relative to a random-article baseline, incorrect retrievals appear strongly skewed toward more popular articles. However, this baseline does not reflect the actual retrieval unit: the index is chunk-based, and popular articles contribute more chunks. When retrievals are instead compared against a random-chunk baseline, the excess preference becomes substantially smaller. This indicates that much of the apparent popularity preference is explained by index exposure rather than by an intrinsic scorer-level preference for popular articles.

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{data/pref-curve.png}
    \caption{Wrong-document popularity preference under chunk-weighted deciles. The curves show the probability that an incorrectly retrieved article is more popular than the target article. The difference between the random-article and random-chunk baselines illustrates the role of index exposure: popular articles occupy more retrieval space because they contribute more chunks.}
    \label{fig:wrong_doc_pref}
\end{figure}


\subsection{Coverage Amplifies the Apparent Gap}

\paragraph{More indexed information creates more retrieval exposure.}
Coverage strongly affects the measured popularity bias. Popular articles are typically longer and therefore contribute more chunks to the retrieval index. As a result, they receive more retrieval opportunities: even if the retriever has no explicit preference for popularity, articles with more indexed content are more likely to appear among the retrieved results.

This explains why incorrect retrievals appear strongly popularity-skewed under an article-level baseline. At the article level, each page counts once, but in the actual chunk index, popular pages may appear many times. When the baseline is changed from random articles to random indexed chunks, much of the apparent preference for popular incorrect documents disappears.

\paragraph{Implications.}
Incorrect retrievals are popularity-skewed at the article level, but this does not imply that retrievers rank documents more highly simply because they are popular. Much of the observed skew is consistent with the greater exposure of popular articles in the chunked index. Consequently, an index dominated by chunks from highly popular articles will also return more highly popular content, even without a direct scorer-level preference. This exposure effect may shape the evidence provided to downstream generation and can therefore amplify popularity bias before generation begins, potentially influencing the final output as strongly as, or more strongly than, the generator itself. This answers RQ2: the popularity-related performance gap is real, but it is not primarily caused by a direct preference for popular documents.

\subsection{Generation Compresses the Retrieval Gap}

\paragraph{Correct retrieval matters most for rare entities.}
Figure~\ref{fig:qwen_gen_retrieval_decile} shows Qwen generation accuracy and retrieval hit rate across popularity deciles. Retrieval quality strongly affects generation: BM25 retrieval accuracy decreases sharply across deciles, and generation accuracy follows the same broad trend. FAISS high remains more stable, while zero-shot accuracy increases toward the highest-popularity deciles. This indicates that retrieval provides the greatest benefit where parametric knowledge is weakest.

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{data/qwen_generation.png}
    \caption{Qwen generation accuracy and retrieval hit rate across chunk-weighted popularity deciles. BM25 retrieval quality decreases strongly with popularity, while zero-shot accuracy increases for popular entities.}
    \label{fig:qwen_gen_retrieval_decile}
\end{figure}

\paragraph{High-popularity entities are less retrieval-bound.}
Figure~\ref{fig:qwen_retrieval_lift_decile} shows the accuracy improvement obtained when the gold article is retrieved. Correct retrieval substantially improves answer accuracy for low- and mid-popularity entities, but this improvement becomes smaller in the highest deciles. Qwen can therefore often answer high-popularity questions using parametric knowledge even when exact retrieval fails.

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{data/qwen_retrieval_lift.png}
    \caption{Qwen answer-accuracy improvement when retrieval contains the gold article. Retrieval hits matter most for lower- and middle-popularity deciles, whereas the benefit decreases for highly popular entities.}
    \label{fig:qwen_retrieval_lift_decile}
\end{figure}

\paragraph{Implications.}
Generation does not eliminate popularity bias; it mainly masks part of the retrieval gap for popular entities. In real-world RAG systems, popular knowledge can often be recovered from the model’s parametric memory even when retrieval fails, whereas rare entities remain highly dependent on retrieving the correct evidence. As a result, retrieval errors continue to disproportionately affect long-tail knowledge, even when aggregate generation accuracy appears more balanced. This also means that evaluating only final answers can underestimate retrieval-stage bias, because the generator may compensate for failures on popular entities while leaving failures on rare entities unresolved.

\section{Limitations}

\paragraph{Wikipedia corpus and exposure.}
Our experiments use Wikipedia, where each entity is typically represented by a single article. This differs from real-world retrieval systems, in which popular entities often have many redundant sources, pages, and mentions. Consequently, Wikipedia may understate the exposure advantage that popular entities receive in production corpora, and the observed effects should not be interpreted as a complete model of real-world information ecosystems.

\paragraph{Dense retrieval analysis.}
The dense-retrieval analysis is limited in scale relative to the sparse BM25 analysis. Computational constraints restrict the number of configurations and diagnostics that can be evaluated, so conclusions about the mechanisms behind dense-retrieval popularity effects are less definitive.

\paragraph{Model coverage.}
We evaluate a limited set of embedding models and generators. The magnitude of the observed effects may vary for larger models, different training data, or models with stronger parametric knowledge. Broader evaluation across retrieval models, LLMs, and retrieval-augmentation settings is needed to establish how general these findings are.

\section{Conclusion}

This thesis shows that popularity-related retrieval disparities arise from more than a direct preference for popular documents. In chunked Wikipedia retrieval, popular articles receive greater exposure because they contribute more indexed content, yet BM25 weakens despite stronger target-side lexical evidence. Dense retrieval is comparatively stronger for popular entities, which is consistent with more separable learned representations.

The downstream consequence is important for real-world RAG systems. Popular facts can often be recovered from a generator's parametric knowledge when retrieval fails, but rare entities remain strongly retrieval-bound. Evaluating only final answer accuracy can therefore hide retrieval failures that disproportionately affect long-tail knowledge. Production systems with many redundant pages, mentions, and sources may amplify this exposure effect beyond what is observed in Wikipedia.

These conclusions are limited by the Wikipedia-only corpus, the smaller scale of the dense-retrieval diagnostics, and the limited set of retrievers and generators evaluated. Nevertheless, the results show that improving equitable RAG requires attention to the retrieval index and retriever, not only to the generator.

\section{Future Work}

Future work should evaluate these effects on web-scale and multi-source corpora, where popular entities have many more redundant documents and mentions than in Wikipedia. This would test whether exposure-driven disparities become larger in realistic search and RAG settings.

A second direction is to test mitigation strategies. Possible approaches include limiting the number of chunks indexed per article, retrieving at the article rather than passage level, diversifying retrieved contexts, and using popularity-aware or exposure-normalized ranking objectives. These interventions should be evaluated for both overall retrieval quality and long-tail performance.

Finally, broader experiments should include more embedding models, rerankers, larger generators, hybrid retrieval strategies, and human evaluation. This would establish whether the observed relationship between popularity, retrieval, and generation generalizes across modern RAG systems.

\begin{thebibliography}{99}

    \bibitem{chowdhery2022palm}
    Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma,
    Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung,
    Charles Sutton, Sebastian Gehrmann, and others. 2022.
    \textit{PaLM: Scaling Language Modeling with Pathways}.
    arXiv preprint arXiv:2204.02311.  [oai_citation:0‡arXiv](https://arxiv.org/abs/2204.02311?utm_source=chatgpt.com)
    
    \bibitem{yu2022}
    Wenhao Yu, Dan Iter, Shuohang Wang, Yichong Xu,
    Mingxuan Ju, Soumya Sanyal, Chenguang Zhu,
    Michael Zeng, and Meng Jiang. 2022.
    \textit{Generate Rather than Retrieve: Large Language Models are Strong Context Generators}.
    arXiv preprint arXiv:2209.10063.
    
    \bibitem{kandpal2022large}
    Nikhil Kandpal, Haikang Deng, Adam Roberts,
    Eric Wallace, and Colin Raffel. 2022.
    \textit{Large Language Models Struggle to Learn Long-Tail Knowledge}.
    arXiv preprint arXiv:2211.08411.
    
    \bibitem{shuster2021retrieval}
    Kurt Shuster, Spencer Poff, Moya Chen,
    Douwe Kiela, and Jason Weston. 2021.
    \textit{Retrieval Augmentation Reduces Hallucination in Conversation}.
    Findings of the Association for Computational Linguistics: EMNLP 2021.
    
    \bibitem{kasai2022temporal}
    Jungo Kasai, Keisuke Sakaguchi, Yoichi Takahashi,
    Ronan Le Bras, Akari Asai, Xinyan Yu,
    Dragomir Radev, Noah A. Smith,
    Yejin Choi, and Kentaro Inui. 2022.
    \textit{Realtime QA: What's the Answer Right Now?}
    
    \bibitem{jang2022knowledge}
    Joel Jang, Seonghyeon Ye, Changho Lee,
    Sohee Yang, Joongbo Shin, Janghoon Han,
    Gyeonghun Kim, and Minjoon Seo. 2022.
    \textit{TemporalWiki: A Lifelong Benchmark for Training and Evaluating Ever-Evolving Language Models}.
    
    \bibitem{liu2024lost}
    Nelson F. Liu, Kevin Lin, John Hewitt,
    Ashwin Paranjape, Michele Bevilacqua,
    Fabio Petroni, and Percy Liang. 2024.
    \textit{Lost in the Middle: How Language Models Use Long Contexts}.
    Transactions of the Association for Computational Linguistics, 12:157--173.  [oai_citation:1‡MIT Press Direct](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00638/119630/Lost-in-the-Middle-How-Language-Models-Use-Long?utm_source=chatgpt.com)
    
    \bibitem{ni2025knowledge}
    Shiyu Ni, Keping Bi, Jiafeng Guo, and Xueqi Cheng. 2025.
    \textit{How Knowledge Popularity Influences and Enhances LLM Knowledge Boundary Perception}.
    arXiv preprint arXiv:2505.17537.
    
    \bibitem{yoshida2015wikipedia}
    Mitsuo Yoshida, Yuki Arase, Takaaki Tsunoda, and Mikio Yamamoto. 2015.
    \textit{Wikipedia Page View Reflects Web Search Trend}.
    In Proceedings of the 2015 ACM Web Science Conference (WebSci '15), Article 64.
    
    \bibitem{mallen2023trust}
    Alex Mallen, Akari Asai, Victor Zhong,
    Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. 2023.
    \textit{When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories}.
    In Proceedings of ACL 2023, pages 9802--9822.
    
    \bibitem{wang2024multilingual}
    Liang Wang, Nan Yang, Xiaolong Huang,
    Linjun Yang, Rangan Majumder, and Furu Wei. 2024.
    \textit{Multilingual E5 Text Embeddings: A Technical Report}.
    arXiv preprint arXiv:2402.05672.
    
    \bibitem{conneau2019unsupervised}
    Alexis Conneau, Kartikay Khandelwal,
    Naman Goyal, Vishrav Chaudhary,
    Guillaume Wenzek, Francisco Guzmán,
    Edouard Grave, Myle Ott,
    Luke Zettlemoyer, and Veselin Stoyanov. 2019.
    \textit{Unsupervised Cross-lingual Representation Learning at Scale}.
    arXiv preprint arXiv:1911.02116.
    
    \bibitem{reimers2019sentence}
    Nils Reimers and Iryna Gurevych. 2019.
    \textit{Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks}.
    arXiv preprint arXiv:1908.10084.
    
    \bibitem{thakur2020augmented}
    Nandan Thakur, Nils Reimers,
    Johannes Daxenberger, and Iryna Gurevych. 2020.
    \textit{Augmented SBERT: Data Augmentation Method for Improving Bi-Encoders for Pairwise Sentence Scoring Tasks}.
    arXiv e-prints.
    
    \bibitem{fb_kilt}
    Fabio Petroni, Aleksandra Piktus, Angela Fan,
    Patrick Lewis, Majid Yazdani, Nicola De Cao,
    James Thorne, Yacine Jernite,
    Vassilis Plachouras, Tim Rocktäschel,
    and Sebastian Riedel. 2020.
    \textit{KILT: A Benchmark for Knowledge Intensive Language Tasks}.
    arXiv preprint arXiv:2009.02252.

    \bibitem{lewis2020rag}
    Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni,
    Vladimir Karpukhin, Naman Goyal, Heinrich Küttler,
    Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel,
    and Douwe Kiela. 2020.
    \textit{Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks}.
    In Advances in Neural Information Processing Systems.
    
    \bibitem{karpukhin2020dense}
    Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis,
    Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020.
    \textit{Dense Passage Retrieval for Open-Domain Question Answering}.
    In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing.
    
    \bibitem{robertson2009bm25}
    Stephen Robertson and Hugo Zaragoza. 2009.
    \textit{The Probabilistic Relevance Framework: BM25 and Beyond}.
    Foundations and Trends in Information Retrieval, 3(4):333--389.
    
    \bibitem{thakur2021beir}
    Nandan Thakur, Nils Reimers, Andreas Rücklé,
    Abhishek Srivastava, and Iryna Gurevych. 2021.
    \textit{BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models}.
    In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks.

    \bibitem{petroni2021kilt}
    Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick S. H. Lewis, Majid Yazdani, Nicola De Cao, James Thorne, Yacine Jernite, Vladimir Karpukhin, Jean Maillard, Vassilis Plachouras, Tim Rockt{\"a}schel, and Sebastian Riedel. 2021.
    \textit{KILT: A Benchmark for Knowledge Intensive Language Tasks}.
    In \textit{Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT 2021)}, pages 2523--2544.
    Association for Computational Linguistics.
    \url{https://www.aclweb.org/anthology/2021.naacl-main.200/}

    \bibitem{black2021gptneo}
    Sid Black, Leo Gao, Phil Wang, Connor Leahy, and Stella Biderman. 2021.
    \textit{GPT-Neo: Large Scale Autoregressive Language Modeling with Mesh-Tensorflow}.
    Zenodo. doi:10.5281/zenodo.5297715.
    
    \bibitem{gao2020pile}
    Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe,
    Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, and others. 2020.
    \textit{The Pile: An 800GB Dataset of Diverse Text for Language Modeling}.
    arXiv preprint arXiv:2101.00027.
    
    \bibitem{yang2024qwen25}
    An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu,
    Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, and others. 2024.
    \textit{Qwen2.5 Technical Report}.
    arXiv preprint arXiv:2412.15115.

    \bibitem{bm25s}
    Xing Han Lù. 2024.
    \textit{BM25S: Orders of Magnitude Faster Lexical Search via Eager Sparse Scoring}.
    arXiv preprint arXiv:2407.03618.
    \url{https://arxiv.org/abs/2407.03618}

    \bibitem{arabzadeh2025benchmarking}
    Negar Arabzadeh and Charles L. A. Clarke. 2025.
    \textit{Benchmarking LLM-based Relevance Judgment Methods}.
    arXiv preprint arXiv:2504.12558.

\end{thebibliography}

\end{document}
