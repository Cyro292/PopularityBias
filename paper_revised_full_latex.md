
\documentclass[11pt,a4paper,twocolumn]{article}

\usepackage[a4paper,margin=1.8cm]{geometry}
\usepackage{microtype}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage[hidelinks]{hyperref}
\usepackage{url}
\usepackage{graphicx}

\title{Popularity, Exposure, and Retrieval Quality in Retrieval-Augmented Generation}
\author{Amon Sisowath}
\date{February 2026}

\begin{document}

\maketitle

\begin{abstract}
Retrieval-Augmented Generation (RAG) is commonly used to compensate for gaps in
the parametric knowledge of language models, particularly for rare or long-tail
facts. However, retrieval itself may vary systematically with entity popularity.
This work studies popularity effects in sparse and dense retrieval over a fixed
2019 Wikipedia corpus. We evaluate BM25 and FAISS-based dense retrieval across
six Wikipedia-linked knowledge-intensive benchmarks, using Wikipedia page views
as an external popularity measure. To separate corpus exposure from retrieval
behavior, we compare article-level and chunk-weighted popularity deciles.

Retrieval performance differs substantially across popularity groups, but the
pattern depends on the retrieval method. BM25 weakens for highly popular targets
despite increasing target-side lexical evidence, consistent with stronger lexical
competition among many plausible chunks. Dense retrieval is comparatively more
stable for popular targets, and a small analogue diagnostic is consistent with
more separable dense representations for older or more historically represented
entities. Wrong-document analysis further shows that much apparent preference for
popular articles is explained by their greater exposure in a chunked index rather
than by a direct scorer-level preference. Finally, generation partially masks
retrieval disparities: popular questions can often be answered from parametric
knowledge when retrieval fails, whereas rare entities remain retrieval-bound.
These results show that equitable RAG requires attention to index construction
and retrieval quality, not only to the generator.
\end{abstract}

\section{Introduction}

Large language models (LLMs) perform strongly on many knowledge-intensive tasks
that require factual knowledge \cite{chowdhery2022palm,yu2022}. Nevertheless,
they can struggle with infrequent and long-tail facts, generate unsupported
statements when knowledge is missing or uncertain, and become outdated as the
world changes. These limitations are especially consequential in factual
question answering and other applications where correctness depends on access to
accurate and current information.

Retrieval-Augmented Generation (RAG) addresses these limitations by retrieving
documents from an external corpus and supplying them as context to a language
model \cite{lewis2020rag}. Retrieval can ground responses in external evidence
and reduce reliance on incomplete parametric memory. It can also reduce the need
to process an entire knowledge source at once, which is useful given the
well-known difficulty of using information placed in the middle of long contexts
\cite{liu2024lost}.

Embedding-based similarity search is central to RAG, but its use extends beyond
retrieval-augmented generation. It also supports entity matching, entity linking,
knowledge graph construction, semantic search, and graph-based retrieval
architectures. Understanding how similarity search behaves across different
types of knowledge is therefore important for a broad class of knowledge
systems.

Retrieval is often presented as a corrective mechanism for weaknesses in
parametric memory. If language models know less about rare entities, retrieval
should be especially useful for long-tail knowledge. The central question is
therefore not only whether rare target articles are retrieved successfully, but
also what replaces them when retrieval fails. Do failures return similarly rare
but related articles, or do they instead return highly popular information from
elsewhere in the corpus?

We distinguish two mechanisms that can produce popularity-skewed retrieval
outputs. \emph{Internal retrieval bias} occurs when a retriever systematically
ranks popular documents ahead of otherwise comparable alternatives after their
availability in the index is considered. \emph{Exposure bias} occurs before
ranking: popular entities often contribute longer articles and more passage
chunks, so they occupy a larger share of a chunked index. A retriever can then
return popular documents more frequently without any internal preference for
popularity.

This distinction matters most for long-tail knowledge. When the correct rare
article is not retrieved, the context passed to the generator may be dominated by
popular but irrelevant evidence simply because popular articles have more
retrieval opportunities. Such exposure-driven errors can still harm long-tail
RAG, even if the ranker is popularity-neutral. Conversely, if wrong retrievals
are more popular than their chunk-level exposure would predict, this would be
evidence for additional internal retrieval bias.

We test these explanations in three steps. First, we compare retrieval
performance across popularity levels. Second, we examine the popularity of wrong
retrievals against random-article and random-chunk baselines. Third, we use
lexical and dense-retrieval diagnostics to identify mechanisms behind the observed
patterns. We then test how retrieval quality affects downstream generation.

We address the following research questions:

\begin{enumerate}
    \item \textbf{RQ1:} How does retrieval performance vary with target popularity?
    \item \textbf{RQ2:} Do wrong retrievals follow the article-level or chunk-level popularity distribution?
    \item \textbf{RQ3:} Which retrieval characteristics are associated with these differences?
\end{enumerate}

We evaluate BM25 and FAISS-based dense retrieval over the 2019 KILT Wikipedia
snapshot \cite{fb_kilt}. Dense retrieval uses three multilingual embedding models:
\texttt{Lajavaness/bilingual-embedding-small},
\texttt{intfloat/multilingual-e5-small}, and
\texttt{intfloat/multilingual-e5-large}. Experiments cover FEVER, HotpotQA,
Natural Questions, PopQA, T-REx, and TriviaQA. Downstream generation uses
GPT-Neo-2.7B and Qwen2.5-7B-Instruct.

Our results have four main implications. First, the dense--BM25 gap grows with
popularity because BM25 weakens while dense retrieval remains comparatively
stable. Second, BM25 degradation is consistent with lexical competition, not
missing target-side lexical evidence. Third, wrong retrievals are much closer to
the random-chunk baseline than to the random-article baseline, showing that
chunk-level exposure explains much of their popularity skew. Fourth, generation
reduces the visible retrieval gap: popular questions are more often recoverable
from parametric knowledge, whereas rare entities remain more retrieval-dependent.
The small analogue diagnostic provides additional, limited-scale evidence
consistent with representation-related dense-retrieval differences.

\section{Related Work}

\paragraph{Parametric and non-parametric knowledge.}
LLMs store substantial factual knowledge in their parameters and can answer many
knowledge-intensive questions without retrieval \cite{chowdhery2022palm,yu2022}.
However, parametric knowledge is incomplete, can become outdated, and can lead to
hallucinated responses without external grounding
\cite{shuster2021retrieval,kasai2022temporal,jang2022knowledge}. Retrieval
augmentation supplies non-parametric knowledge by conditioning generation on
documents from an external corpus \cite{lewis2020rag}.

\paragraph{Popularity and factual memorization.}
Factual knowledge is not learned uniformly by language models. Kandpal et
al.~\cite{kandpal2022large} show that long-tail factual knowledge is difficult
for language models to learn. Mallen et al.~\cite{mallen2023trust} find that
entity popularity predicts whether models can answer factual questions from
parametric memory: popular entities are answered more reliably, whereas less
popular entities more often require retrieval. These findings motivate studying
whether the retrieval component itself also varies with popularity.

\paragraph{Retrieval augmentation and retrieval quality.}
RAG improves factual answering by supplying external evidence
\cite{lewis2020rag,shuster2021retrieval}. Mallen et al.~\cite{mallen2023trust}
show that retrieval is especially useful when parametric knowledge is weak.
However, prior work primarily studies the interaction between parametric and
non-parametric memory. It does not fully separate corpus exposure, ranking
behavior, and representation quality as sources of popularity-related retrieval
differences.

\paragraph{Positioning of this work.}
This work studies popularity directly at the retrieval stage. Rather than treating
all popularity effects as a single bias, it distinguishes exposure in a chunked
index, direct preference for popular documents, lexical discriminability, and
dense representation quality. This distinction is necessary because each
mechanism implies a different intervention.

\section{Evaluation Setup}

We evaluate an open-domain RAG pipeline over a fixed Wikipedia corpus. Given a
question, a retriever searches a full index of Wikipedia passage chunks and
returns a ranked list. The top-ranked passages can then be passed to a generator
as context. We evaluate both article-level retrieval quality and downstream
answer accuracy.

\subsection{Task Definition}

Questions are linked to target Wikipedia articles, while the corpus consists of
passage chunks. A retrieved passage is therefore considered relevant if its
metadata points to the target Wikipedia article. Retrieval is evaluated at the
article level: the retriever need only return at least one chunk from the target
article, rather than a particular annotated span.

We report Recall@1, Recall@3, Recall@5, Recall@10, and mean reciprocal rank
(MRR); the main decile figures use Recall@10. Recall@10 is defined as

\begin{equation}
    \mathrm{Recall@10}(q) =
    \mathbb{I}\left[\exists d \in R_{10}(q): \mathrm{wiki}(d) = a_q\right],
\end{equation}

where $R_{10}(q)$ is the top ten retrieved chunks for question $q$, and $a_q$
is its target article. MRR uses the rank of the first chunk associated with the
target article; the contribution is zero if no target chunk is retrieved.

For downstream generation, the top three passages are concatenated and supplied
to the generator. We report case-insensitive substring accuracy, which marks an
answer as correct when it contains an accepted reference answer. We additionally
use a binary LLM-based judge where results are available. The judge receives the
question, generated answer, and reference information and decides whether the
answer is correct. This provides a robustness check for cases where correct
answers differ lexically from the reference \cite{arabzadeh2025benchmarking}.

\subsection{Popularity and Exposure}

We use Wikipedia page views as an external measure of target-article popularity,
following Mallen et al.~\cite{mallen2023trust}. Specifically, we use monthly
page views averaged from 2020 through 2024 to reduce sensitivity to short-lived
news events and viral spikes. Page views are model-independent, available for
the corpus articles, and provide a practical proxy for public visibility. They
do not measure a model's training frequency directly.

Article-level popularity deciles treat every article equally. This can be
misleading for passage retrieval because popular articles are often longer and
contribute more chunks to the index. We therefore also compute chunk-weighted
popularity deciles, where articles are weighted by their number of indexed
chunks when decile boundaries are formed. Each chunk-weighted decile consequently
contains approximately equal indexed mass.

Chunk-weighting reduces the exposure channel caused by unequal index mass across
popularity groups. It does not equalize article length, target difficulty, topic,
or semantic distinctiveness within a decile. Results under chunk-weighted deciles
should therefore be interpreted as less exposure-confounded comparisons, not as
fully controlled causal estimates.

\subsection{Benchmarks}

We evaluate Wikipedia-linked knowledge-intensive datasets to avoid dependence on
a single query style. PopQA is an entity-centric open-domain QA dataset designed
to cover a broad popularity range \cite{mallen2023trust}. Natural Questions
contains real user questions paired with Wikipedia evidence, while TriviaQA adds
questions collected from trivia sources \cite{petroni2021kilt}. FEVER contributes
Wikipedia-grounded fact-verification claims. HotpotQA contains multi-hop
questions, and T-REx provides entity-relation queries derived from Wikidata and
aligned with Wikipedia \cite{petroni2021kilt}.

After linking examples to the 2019 Wikipedia snapshot and filtering unavailable
targets, we construct two pools. The smaller pool supports repeated retrieval
runs and diagnostics; the larger pool supports broader retrieval estimates.
Questions are sampled approximately evenly across popularity deciles, with
downsampling performed using a fixed random seed of 42.

\begin{table}[t]
\centering
\small
\begin{tabular}{lrr}
\toprule
\textbf{Dataset} & \textbf{8k pool} & \textbf{60k pool} \\
\midrule
FEVER & 1,064 & 5,451 \\
HotpotQA & 1,325 & 9,675 \\
Natural Questions & 1,212 & 6,403 \\
PopQA & 1,330 & 9,702 \\
T-REx & 1,329 & 9,953 \\
TriviaQA & 1,123 & 6,007 \\
\midrule
\textbf{Total} & \textbf{7,383} & \textbf{47,191} \\
\bottomrule
\end{tabular}
\caption{Question distribution after Wikipedia linking and filtering.}
\label{tab:question_distribution}
\end{table}

\subsection{Retrievers}

We compare sparse, dense, and hybrid retrieval. Sparse retrieval tests lexical
matching, dense retrieval tests embedding-based semantic similarity, and hybrid
retrieval combines sparse and dense rankings with reciprocal-rank fusion.

For sparse retrieval, we use \texttt{bm25s} \cite{bm25s}. The main sparse
variant is \texttt{bm25\_plus}, with $k_1=1.5$, $b=0.75$, and $\delta=1.0$.

For dense retrieval, we evaluate multilingual sentence embeddings from the MTEB
ecosystem:
\begin{itemize}
    \item \texttt{Lajavaness/bilingual-embedding-small}, 384 dimensions \cite{conneau2019unsupervised,reimers2019sentence,thakur2020augmented};
    \item \texttt{intfloat/multilingual-e5-small}, 384 dimensions \cite{wang2024multilingual};
    \item \texttt{intfloat/multilingual-e5-large}, 1024 dimensions \cite{wang2024multilingual}.
\end{itemize}

The 2019 corpus contains 5,903,530 articles, of which 5,890,044 have popularity
metadata. With 1,000-character chunks and 100-character overlap, this produces
24,651,978 indexed chunks. Dense indexes are implemented with FAISS IVF-PQ. The
implementation uses 4,096 inverted lists, 48 subquantizers, and 8-bit codes; the
reported \texttt{ivfpq\_high} setting uses normalized
\texttt{bilingual-embedding-small} vectors and \texttt{nprobe=256} at query
time. The embedding service applies the default templates
\texttt{query: \{query\}} and \texttt{passage: \{passage\}} consistently to
dense queries and passages.

\subsection{Generators}

The main RAG experiments use the top three retrieved passages. The prompt is
kept fixed across retrieval systems:

\begin{quote}
Documents: \{documents\}

Question: \{question\}
\end{quote}

The zero-shot condition omits retrieved passages and measures how often a
generator answers from parametric knowledge alone. We evaluate GPT-Neo-2.7B
(\texttt{EleutherAI/gpt-neo-2.7B}) \cite{black2021gptneo} and Qwen2.5-7B-Instruct
(\texttt{Qwen/Qwen2.5-7B-Instruct}) \cite{yang2024qwen25}.

\subsection{Analogue Similarity Diagnostic}

To probe whether dense retrieval scores may reflect representation strength
beyond lexical overlap, we construct an approximate diagnostic set of 50 entity
pairs. Each pair contains structurally similar Wikipedia entities from the same
broad category: a newer or less historically represented entity from the 2026
Wikipedia snapshot and an older, more popular analogue. Matched queries express
the same relation where possible.

This diagnostic is not a benchmark and does not identify a causal effect of
popularity. Entity age, article quality, topic, and training exposure may remain
confounded. We compare dense query--target similarity with BM25 as a lexical
control. A larger dense difference when BM25 remains similar is evidence
consistent with, but not proof of, a representation-strength mechanism.

\section{Findings}

We examine whether retrieval performance changes with target popularity and
whether the observed differences are associated with coverage, lexical
competition, dense representation quality, or direct popularity preference.

\subsection{Retrieval Performance Depends on Popularity}
\paragraph{Popularity widens the dense--BM25 retrieval gap.}

Figure~\ref{fig:per_decile_retrieval} shows Recall@10 across chunk-weighted popularity deciles. The dominant trend is not a uniform popularity advantage across retrieval methods. Instead, retrieval systems respond differently to increasing popularity. Dense retrieval remains stable or improves, whereas BM25 performance steadily declines. Consequently, the performance gap between sparse and dense retrieval widens toward the higher-popularity deciles.

This trend is consistent across all evaluated retrieval backends. Figure~\ref{fig:delta_vs_bm25} illustrates the increasing dense--BM25 performance difference for the representative \texttt{bm25\_plus} and \texttt{ivfpq\_high} configurations, while the complete results tables show that the same qualitative pattern holds across all sparse and dense retrievers considered in this study. These findings answer RQ1: popularity influences retrieval performance, but the effect depends strongly on the retrieval architecture.

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{data/recall10_pop.png}
    \caption{Recall@10 across chunk-weighted popularity deciles. Dense retrieval remains stable or improves for popular targets, whereas BM25 performance steadily declines.}
    \label{fig:per_decile_retrieval}
\end{figure}

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{data/delta_vs_bm25_retrieved_docs_ivfpq_high.png}
    \caption{Difference in Recall@10 between dense retrieval and BM25 by dataset and chunk-weighted popularity decile. The dense retrieval advantage increases with popularity.}
    \label{fig:delta_vs_bm25}
\end{figure}

\paragraph{BM25 failures primarily arise during ranking.}

The decline of BM25 is initially surprising because popularity simultaneously strengthens several factors that should improve lexical retrieval. Popular articles contain substantially more chunks, increasing the number of opportunities to retrieve the correct article under our article-level evaluation, where every target chunk counts as correct. In addition, the best target chunks exhibit greater query-term overlap, higher term frequency, higher TF--IDF evidence, and a larger fraction of matching chunks. Nevertheless, BM25+ Hit@1 decreases from 43.8\% in the lowest chunk-weighted popularity decile to 15.4\% in the highest, a relative decrease of 64.7\%.

Figure~\ref{fig:bm25_degen} shows that these improvements in target-side evidence are accompanied by only a moderate reduction in average query-term IDF (5.15 to 4.76). While lower IDF reduces the discriminative power of query terms, this effect is opposed by substantially stronger lexical evidence and increased target exposure. Taken together, these diagnostics suggest that neither weaker target-side evidence nor reduced query specificity alone can explain the observed decline.

Direct analysis of BM25 scores identifies where retrieval fails. At retrieval depth 100, the correct article is retrieved at nearly identical rates in the lowest and highest popularity deciles (80.1\% versus 78.6\%; $p=0.57$), indicating that candidate generation remains largely unaffected. The dominant failure instead occurs during ranking. Conditional on the target article being retrieved, its mean rank deteriorates from 6.13 to 13.95, while the score margin between the highest-scoring target chunk and the strongest competing chunk changes from $+1.32$ to $-1.41$ ($p<0.001$). At the same time, the number of competing chunks within 5\% of the best target score increases from 19.4 to 49.6 ($p<0.001$). These findings indicate that distinguishing the correct article from competing passages becomes substantially more difficult for popular entities.

One important contributor appears to be reduced discriminability of entity-identifying terms. Although queries mention the target title equally often across popularity deciles (87.7\% versus 87.3\%), the cumulative BM25 IDF contribution of the shared title terms decreases from 13.84 to 9.46 (31.6\%; $p<0.001$). Popular entities therefore remain explicitly referenced in the query, but their identifying terms become substantially less informative because they occur in many more documents. Consequently, many non-target passages receive BM25 scores that are nearly indistinguishable from the correct article.

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{data/bm25_failure_factors_retrieved_docs_bm25_plus.png}
    \caption{Lexical diagnostics across chunk-weighted popularity deciles. Target-side lexical evidence generally increases with popularity, while BM25 retrieval performance declines. Direct score analysis shows that popular entities exhibit smaller target--competitor score margins and substantially more near-tied competing chunks. Error bars denote 95\% confidence intervals.}
    \label{fig:bm25_degen}
\end{figure}

\paragraph{Implications.}

These findings suggest that popularity affects BM25 primarily through ranking rather than candidate generation. Greater index exposure and stronger target-side lexical evidence are insufficient to offset the reduced discriminability of popular entity terms and the increasing number of near-equally scoring competing passages. As a result, popular entities remain present in the candidate set but are increasingly displaced by lexically similar alternatives during ranking.

\subsection{Dense Retrieval Is Comparatively Stronger for Popular Knowledge}

\paragraph{Analogue diagnostic.}
Figure~\ref{fig:similarity_analysis} reports the analogue similarity diagnostic.
Dense similarity is higher for the older or more popular analogue entities,
whereas the corresponding BM25 differences are not statistically significant.
Given the matched query structures, this is consistent with stronger or more
separable dense representations for historically represented entities.

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{data/similarity_analysis.png}
    \caption{Similarity analysis over manually selected analogue pairs. Dense
    similarity is higher for older and more popular analogues, whereas the BM25
    difference is not statistically significant.}
    \label{fig:similarity_analysis}
\end{figure}

This result should be interpreted cautiously. The diagnostic contains only about
50 manually selected pairs and cannot separate popularity from entity age,
article quality, or unobserved training exposure. It supports RQ3 as suggestive
evidence for a representation-related mechanism, rather than establishing that
popularity itself causes stronger embeddings.

\paragraph{Implications.}
Popularity-related effects may not be limited to generation. Dense similarity
systems used in semantic search, entity matching, entity linking, and graph-based
retrieval may reproduce related disparities if less represented entities are
embedded less distinctively. This motivates broader evaluation of representation
quality across long-tail knowledge.

\subsection{Popularity Effects Are Not Necessarily Direct Preference}

\paragraph{Wrong-document analysis separates skew from preference.}
Performance differences by target popularity do not establish that a retriever
directly favors popular documents. We therefore analyze only incorrect
retrievals, comparing the popularity of the target article with the popularity
of the incorrectly retrieved article. For target decile $d$, we compute

\begin{equation}
    \mathrm{Pref}(d) =
    \frac{1}{|E_d|}
    \sum_{(q,r) \in E_d}
    \mathbb{I}\left[\mathrm{pop}(r) > \mathrm{pop}(a_q)\right],
\end{equation}

where $E_d$ is the set of incorrect retrievals for questions whose target article
$a_q$ lies in decile $d$, and $r$ is an incorrectly retrieved article.

Figure~\ref{fig:wrong_doc_pref} shows the observed preference alongside random
article and random chunk baselines. Relative to a random-article baseline,
incorrect retrievals are strongly skewed toward more popular articles. This
baseline, however, does not reflect the retrieval unit. The index is chunk-based
and popular articles contribute more chunks. Under a random-chunk baseline, the
excess skew is substantially smaller. In other words, wrong retrievals are much
closer to the distribution induced by random indexed chunks than to a distribution
in which every article is equally likely. A chunked corpus with disproportionate
mass from popular articles can therefore return popular wrong documents more
often, even without direct scorer-level preference.

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{data/pref-curve.png}
    \caption{Wrong-document popularity preference under chunk-weighted deciles.
    The random-article and random-chunk baselines show how chunk-level index
    exposure contributes to the apparent popularity skew.}
    \label{fig:wrong_doc_pref}
\end{figure}

\paragraph{Implications.}
The curve separates \emph{exposure bias} from \emph{internal retrieval bias}.
The observed errors are much closer to the random-chunk baseline than to the
random-article baseline. This means that a substantial share of the popularity
skew is exposure bias: popular articles occupy more chunk-level retrieval space,
so they appear more often among wrong results even without a direct ranker
preference. The remaining difference from the chunk baseline is the part that
could reflect additional internal retrieval bias.

For long-tail knowledge, this is consequential. When retrieval misses a rare
target, the replacement context is not distributed as if every article had an
equal chance of being returned. It is disproportionately drawn from the more
popular content that dominates the chunked index. Thus, rare targets are not
unaffected by exposure imbalance: retrieval failures can replace their evidence
with popular but irrelevant context before generation begins. The current curve
measures the popularity of this replacement context, not its semantic proximity
to the target. Determining whether the wrong articles are close long-tail
neighbours or distant popular distractors requires a separate similarity or
topic-overlap analysis.

\subsection{Generation Compresses the Retrieval Gap}

\paragraph{Correct retrieval matters most for rare entities.}
Figure~\ref{fig:qwen_gen_retrieval_decile} shows Qwen generation accuracy and
retrieval hit rate across popularity deciles. BM25 retrieval accuracy decreases
sharply across deciles, and generation accuracy follows the same broad pattern.
FAISS high is more stable, while zero-shot accuracy rises toward the most popular
deciles. This pattern is consistent with retrieval being most valuable where
Qwen's parametric knowledge is weakest.

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{data/qwen_generation_retrieval_accuracy_by_decile.png}
    \caption{Qwen generation accuracy and retrieval hit rate across
    chunk-weighted popularity deciles. BM25 retrieval quality decreases with
    popularity, whereas zero-shot generation improves for popular entities.
    Vertical bars show 95\% normal confidence intervals.}
    \label{fig:qwen_gen_retrieval_decile}
\end{figure}

\paragraph{High-popularity entities are less retrieval-bound.}
Figure~\ref{fig:qwen_retrieval_lift_decile} shows the conditional difference in
answer accuracy between retrieval hits and misses. Correct retrieval is
associated with substantially higher answer accuracy for low- and
mid-popularity entities, but this difference decreases in the highest deciles.
This is consistent with Qwen answering many high-popularity questions from
parametric knowledge even when exact retrieval fails.

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{data/qwen_retrieval_hit_lift_by_decile.png}
    \caption{Qwen conditional answer-accuracy difference between retrieval hits
    and misses. The association is largest for lower and middle popularity
    deciles and decreases for highly popular entities. Vertical bars show 95\%
    normal confidence intervals.}
    \label{fig:qwen_retrieval_lift_decile}
\end{figure}

The hit--miss comparison is observational: questions that retrieve successfully
may also be easier for other reasons. It should not be read as a causal estimate
of the benefit of retrieval. Nevertheless, together with the zero-shot trend, it
shows that rare entities remain more dependent on retrieval than popular ones.
The same qualitative relationship holds for both tested generators and across
the evaluated retrieval backends. Neo is not shown separately in the main figure
only to keep the presentation readable; the accompanying result tables report
both generators and all tested configurations.

\paragraph{Binary-evaluation robustness check.}
The same popularity-decile analysis is configured for the binary LLM evaluator.
Once the binary result files are added, it will report the same Qwen generation
and retrieval-accuracy plot together with the conditional hit--miss comparison.
The binary figures are included as a robustness check of the substring-based
results; they are not used to make an additional mechanism claim.

% Add when binary evaluation results are available:
% \begin{figure}[t]
%     \centering
%     \includegraphics[width=\linewidth]{data/qwen_generation_retrieval_accuracy_by_decile_binary_mistral.png}
%     \caption{Binary-evaluator version of the Qwen popularity-decile analysis.}
%     \label{fig:qwen_binary_gen_retrieval_decile}
% \end{figure}

\paragraph{Implications.}
Generation does not eliminate popularity effects; it compresses them. In
real-world RAG systems, popular facts can often be recovered from parametric
memory after retrieval failure, whereas rare entities remain strongly dependent
on correct evidence. Evaluating only final answer accuracy can therefore
underestimate retrieval-stage disparities, because popular retrieval failures
are easier for the generator to repair.

\section{Limitations}

\paragraph{Wikipedia corpus and exposure.}
Wikipedia usually represents an entity with one primary article. Real-world
retrieval systems often contain many redundant sources, pages, snippets, and
mentions for popular entities. Wikipedia may therefore understate the exposure
advantage that popular entities receive in production corpora. The results should
not be interpreted as a complete model of real-world information ecosystems.

\paragraph{Dense-retrieval diagnostic.}
The dense analogue analysis is small and manually constructed. It confounds
popularity with entity age, historical training exposure, article quality, and
other unobserved properties. It is useful as a diagnostic, but it cannot
establish a population-level causal mechanism for dense retrieval.

\paragraph{Model and configuration coverage.}
Only a limited set of embedding models, retrieval configurations, and generators
is evaluated. Results may differ for modern rerankers, larger generators,
differently trained embedding models, alternative chunking schemes, and other
retrieval corpora. The binary generation evaluation should also be extended and
reported consistently across all configurations.

\paragraph{Article-level relevance and generation attribution.}
Article-level relevance treats any chunk from a linked target article as correct.
This is practical for the current corpus but may not capture whether the retrieved
passage contains the exact answer-bearing evidence, especially for multi-hop or
multi-evidence questions. Similarly, hit--miss generation comparisons are
associational and do not isolate the causal effect of retrieved context.

\section{Conclusion}

This thesis identifies two distinct popularity-related retrieval effects. First,
dense retrieval exhibits a popularity-related performance advantage: it is
comparatively stronger for popular targets, while the limited-scale analogue
diagnostic is consistent with more separable representations for historically
represented entities. BM25 follows a different pattern, weakening for popular
targets as lexical competition increases. Second, retrieval errors show an
exposure-driven preference for popular content. Wrong results are much closer to
the random-chunk baseline than to the random-article baseline, meaning that
popular articles are returned more often because they occupy more of the chunked
index.

The practical consequence concerns long-tail reliability. When retrieval misses a
rare target, the replacement context is disproportionately drawn from popular
content. Popular questions can often still be answered from a generator's
parametric knowledge, but rare entities remain dependent on retrieving the right
evidence. A RAG system can therefore appear balanced when evaluated only by final
answer accuracy while still failing to retrieve evidence for the long tail.
Improving long-tail RAG requires attention to index construction and retrieval
quality, not only to the generator.

\section{Future Work}

Future work should evaluate popularity effects on web-scale and multi-source
corpora, where popular entities have many more redundant pages, mentions, and
aliases than in Wikipedia. This would test whether exposure-driven disparities
increase in realistic search and production RAG settings.

A second direction is direct mitigation. Candidate interventions include limiting
the number of indexed chunks per article, retrieving or aggregating at the article
level, diversifying retrieved contexts, using adaptive retrieval depth, and
optimizing exposure-normalized or long-tail-aware ranking objectives. These
methods should be evaluated jointly for average retrieval quality and for
performance on rare entities.

Finally, future studies should evaluate more embedding models, cross-encoder or
LLM rerankers, stronger generators, and larger binary or human evaluation sets.
A causal generation experiment could compare the same question under no context,
incorrect context, retrieved target context, and oracle target context. This
would isolate how retrieval failure contributes to the downstream long-tail gap.

\begin{thebibliography}{99}

\bibitem{arabzadeh2025benchmarking}
Negar Arabzadeh and Charles L. A. Clarke. 2025. \textit{Benchmarking LLM-based Relevance Judgment Methods}. arXiv preprint arXiv:2504.12558.

\bibitem{black2021gptneo}
Sid Black, Leo Gao, Phil Wang, Connor Leahy, and Stella Biderman. 2021. \textit{GPT-Neo: Large Scale Autoregressive Language Modeling with Mesh-Tensorflow}. Zenodo. doi:10.5281/zenodo.5297715.

\bibitem{bm25s}
Xing Han Lu. 2024. \textit{BM25S: Orders of Magnitude Faster Lexical Search via Eager Sparse Scoring}. arXiv preprint arXiv:2407.03618.

\bibitem{chowdhery2022palm}
Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, and others. 2022. \textit{PaLM: Scaling Language Modeling with Pathways}. arXiv preprint arXiv:2204.02311.

\bibitem{conneau2019unsupervised}
Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzman, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020. \textit{Unsupervised Cross-lingual Representation Learning at Scale}. In Proceedings of ACL 2020.

\bibitem{fb_kilt}
Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Majid Yazdani, Nicola De Cao, James Thorne, Yacine Jernite, Vassilis Plachouras, Tim Rocktaschel, and Sebastian Riedel. 2020. \textit{KILT: A Benchmark for Knowledge Intensive Language Tasks}. arXiv preprint arXiv:2009.02252.

\bibitem{jang2022knowledge}
Joel Jang, Seonghyeon Ye, Changho Lee, Sohee Yang, Joongbo Shin, Janghoon Han, Gyeonghun Kim, and Minjoon Seo. 2022. \textit{TemporalWiki: A Lifelong Benchmark for Training and Evaluating Ever-Evolving Language Models}. In Proceedings of EMNLP 2022.

\bibitem{kandpal2022large}
Nikhil Kandpal, Haikang Deng, Adam Roberts, Eric Wallace, and Colin Raffel. 2022. \textit{Large Language Models Struggle to Learn Long-Tail Knowledge}. arXiv preprint arXiv:2211.08411.

\bibitem{kasai2022temporal}
Jungo Kasai, Keisuke Sakaguchi, Yoichi Takahashi, Ronan Le Bras, Akari Asai, Xinyan Yu, Dragomir Radev, Noah A. Smith, Yejin Choi, and Kentaro Inui. 2022. \textit{REALTIME QA: What's the Answer Right Now?} In Advances in Neural Information Processing Systems.

\bibitem{lewis2020rag}
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler, Mike Lewis, Wen-tau Yih, Tim Rocktaschel, Sebastian Riedel, and Douwe Kiela. 2020. \textit{Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks}. In Advances in Neural Information Processing Systems.

\bibitem{liu2024lost}
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. \textit{Lost in the Middle: How Language Models Use Long Contexts}. Transactions of the Association for Computational Linguistics, 12:157--173.

\bibitem{mallen2023trust}
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. 2023. \textit{When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories}. In Proceedings of ACL 2023, pages 9802--9822.

\bibitem{petroni2021kilt}
Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick S. H. Lewis, Majid Yazdani, Nicola De Cao, James Thorne, Yacine Jernite, Vladimir Karpukhin, Jean Maillard, Vassilis Plachouras, Tim Rocktaschel, and Sebastian Riedel. 2021. \textit{KILT: a Benchmark for Knowledge Intensive Language Tasks}. In Proceedings of NAACL-HLT 2021, pages 2523--2544.

\bibitem{reimers2019sentence}
Nils Reimers and Iryna Gurevych. 2019. \textit{Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks}. In Proceedings of EMNLP-IJCNLP 2019.

\bibitem{shuster2021retrieval}
Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela, and Jason Weston. 2021. \textit{Retrieval Augmentation Reduces Hallucination in Conversation}. In Findings of EMNLP 2021.

\bibitem{thakur2020augmented}
Nandan Thakur, Nils Reimers, Johannes Daxenberger, and Iryna Gurevych. 2021. \textit{Augmented SBERT: Data Augmentation Method for Improving Bi-Encoders for Pairwise Sentence Scoring Tasks}. In Proceedings of NAACL 2021.

\bibitem{wang2024multilingual}
Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. 2024. \textit{Multilingual E5 Text Embeddings: A Technical Report}. arXiv preprint arXiv:2402.05672.

\bibitem{yang2024qwen25}
An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, and others. 2024. \textit{Qwen2.5 Technical Report}. arXiv preprint arXiv:2412.15115.

\bibitem{yu2022}
Wenhao Yu, Dan Iter, Shuohang Wang, Yichong Xu, Mingxuan Ju, Soumya Sanyal, Chenguang Zhu, Michael Zeng, and Meng Jiang. 2022. \textit{Generate Rather than Retrieve: Large Language Models are Strong Context Generators}. arXiv preprint arXiv:2209.10063.

\end{thebibliography}

\end{document}
