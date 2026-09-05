# Advanced RAG Notes: RAG Fundamentals

These revision notes cover:

- **Lesson 1.1:** Why RAG Is Important
- **Lesson 1.2:** RAG and Its Components

The notes explain how an LLM uses parametric knowledge, why RAG is needed, the major components and phases of a RAG system, common failure points, and the difference between RAG and fine-tuning.

## Learning Objectives

After completing this chapter, you should be able to explain:

1. How an LLM uses parametric knowledge to generate a response.
2. Why an LLM cannot depend only on its internal knowledge.
3. What RAG is and why it is needed.
4. The responsibility of every major RAG component.
5. The difference between indexing and query-time processing.
6. The difference between retriever failure and generator failure.
7. When to use RAG, prompt engineering, or fine-tuning.

---

## 1. How an LLM Generates a Response

A Large Language Model, or LLM, is a neural network trained on a very large amount of text data. During training, it learns language patterns, concepts, and relationships between words and ideas.

These learned representations are encoded in the model's weights or parameters. The knowledge represented by those parameters is called **parametric knowledge**.

```text
Large Training Dataset
        ↓ Training
Model Parameters or Weights
        ↓
Parametric Knowledge
        ↓
User Query → Inference → Generated Response
```

An LLM does not store its complete training dataset like a conventional database. It does not normally search for and copy a stored row when it receives a query. Instead, it uses learned patterns and the tokens already present in the input to predict the next token repeatedly.

```text
LLM response = Parametric knowledge + Supplied context + User query
```

### Key point

> Model parameters contain learned representations and patterns, not a directly searchable copy of the complete training dataset.

---

## 2. Why RAG Is Important

LLMs are powerful, but relying only on parametric knowledge creates four major limitations.

### 2.1 Knowledge Cutoff

An LLM learns from the data available during its training process. It does not automatically know information that was created or changed after that training period.

Examples include:

- a recently updated company policy;
- the current price of a product;
- recent news or events;
- a newly released version of software documentation.

Retraining a model whenever information changes would be expensive and time-consuming. RAG can instead retrieve the latest information from an external knowledge source at runtime.

### 2.2 Hallucination

When an LLM does not have sufficient or accurate information, it may still generate a confident and plausible-looking answer. If that answer is false or unsupported, it is called a **hallucination**.

For example, a model may invent a refund rule when it does not have access to the real refund policy.

RAG reduces this risk by retrieving relevant information from a trusted knowledge base and adding it to the model's context.

However:

> RAG reduces hallucination; it does not eliminate hallucination completely.

Incorrect retrieval, conflicting documents, weak prompts, or poor generation can still produce an unreliable answer.

### 2.3 No Source Attribution

When an answer is generated only from parametric knowledge, the application usually cannot identify the exact document from which a specific fact was learned.

RAG can preserve metadata such as:

```python
{
    "source": "employee-policy.pdf",
    "page": 12,
    "section": "Leave Policy"
}
```

The application can use this metadata to display filenames, page numbers, document sections, or URLs with the final answer.

### 2.4 No Private-Data Access

A general-purpose LLM cannot automatically access a company's internal documents, proprietary databases, or a user's private files.

RAG connects the application to an authorized external knowledge base. It can therefore retrieve private information at runtime without permanently adding that information to the model's parameters.

Example: A project-onboarding assistant can retrieve internal architecture documents and answer questions for a newly joined employee.

---

## 3. What Is RAG?

**RAG stands for Retrieval-Augmented Generation.**

RAG is a technique in which an application retrieves information relevant to a user's query from an external knowledge base, adds that information to the prompt, and asks an LLM to generate a grounded response.

```text
User Query
    ↓
Retrieve Relevant Information
    ↓
Query + Retrieved Context
    ↓
LLM
    ↓
Grounded Answer + Sources
```

The three parts of the term describe the complete process:

1. **Retrieval:** Find information relevant to the user's query.
2. **Augmentation:** Add the retrieved information to the model's input prompt.
3. **Generation:** Use the LLM to generate the final response.

Standard RAG does not modify the parameters of the LLM. External knowledge is supplied at inference time.

### Practical example

Suppose a user asks:

```text
How many casual-leave days are employees allowed each year?
```

The application may perform these steps:

```text
1. Search employee-policy.pdf.
2. Retrieve the relevant Leave Policy section.
3. Add that section to the prompt as context.
4. Ask the LLM to answer using the supplied context.
5. Display the answer with the filename and page number.
```

---

## 4. In-Context Learning and the Context Window

### 4.1 In-Context Learning

**In-Context Learning**, or ICL, means providing instructions, examples, or external knowledge inside the prompt so that the model can use them without additional training.

```text
Instructions + External Context + User Query
                      ↓
                     LLM
                      ↓
                   Response
```

The information supplied in the prompt does not permanently change the model's parameters.

### 4.2 Why Not Put Every Document in the Prompt?

Every LLM has a limited **context window**. It determines how many tokens the model can process in one request.

```text
Context Window =
System Instructions
+ Conversation History
+ Retrieved Documents
+ User Query
+ Space for the Generated Answer
```

Providing every document directly in the prompt can cause several problems:

- The input may exceed the context-window limit.
- Token cost increases.
- Response latency increases.
- Irrelevant information can distract the model.
- Conflicting passages may make the answer less reliable.
- Important evidence may become difficult to identify within a large input.
- Less token space remains available for the generated answer.

RAG addresses this problem by selecting only the information that is most relevant to the current query.

---

## 5. Core Components of RAG

| Component | Responsibility | Output |
|---|---|---|
| Knowledge Source | Provides external information | Raw files or records |
| Document Loader | Reads and standardizes source data | Documents |
| Text Splitter | Divides large documents into smaller units | Chunks |
| Embedding Model | Creates semantic numerical representations | Vectors |
| Vector Store | Stores and searches vectors, text, and metadata | Similar chunks |
| Retriever | Selects context relevant to the query | Retrieved documents |
| Prompt Builder | Combines instructions, context, and query | Augmented prompt |
| Generator or LLM | Generates a grounded natural-language answer | Final response |

### 5.1 Knowledge Source

A knowledge source contains the external information that the RAG system can search.

Examples include:

- PDF, TXT, Markdown, and Word files;
- websites and product documentation;
- CSV and JSON data;
- SQL and NoSQL databases;
- cloud-storage objects;
- internal company knowledge bases.

### 5.2 Document Loader

A document loader reads content from a source and converts it into a standardized document representation.

For example, LangChain uses `Document` objects:

```python
Document(
    page_content="The actual text of the document",
    metadata={
        "source": "policy.pdf",
        "page": 5
    }
)
```

- `page_content` contains the actual text.
- `metadata` contains supporting information such as the source, page, title, date, or URL.

### 5.3 Text Splitter or Chunker

Large documents are divided into smaller searchable units called **chunks**.

```text
Large Document → Chunk 1 + Chunk 2 + Chunk 3 + ...
```

Smaller chunks help the retriever locate a specific relevant passage. However, chunks that are too small may lose the surrounding context needed to answer the question.

Important distinction:

> A document loader reads and standardizes data. A text splitter divides that data into chunks.

### 5.4 Embedding Model

An embedding model converts text into a fixed-length numerical vector that represents semantic meaning.

```text
Document Chunk → Embedding Model → Document Vector
User Query     → Embedding Model → Query Vector
```

Semantically related text should produce vectors that are close to one another in the embedding space.

The document chunks and user query should normally be embedded using the same embedding model so that their vectors exist in the same vector space.

### 5.5 Vector Store

A vector store saves document embeddings along with their original text and metadata. It can compare a query vector with stored vectors and return the closest matches.

Typical operations include:

- adding vectors and documents;
- performing similarity search;
- updating or deleting indexed data;
- returning matching text and metadata.

The original text is needed for generation, while metadata is needed for filtering, tracing, and source attribution.

### 5.6 Retriever

A retriever accepts a query and returns relevant documents or chunks from the knowledge base.

```text
User Query → Query Vector → Similarity Search → Relevant Chunks
```

The retriever finds context; it does not generate the final answer.

Common retrieval strategies include:

- similarity search;
- similarity-score threshold;
- Maximum Marginal Relevance, or MMR;
- BM25 keyword search;
- hybrid search;
- reranking.

### 5.7 Prompt Augmentation

The retrieved context, user query, and instructions are combined to create an augmented prompt.

```text
Instructions + Retrieved Context + User Query = Augmented Prompt
```

A strong RAG prompt commonly tells the model to:

- answer using the supplied context;
- avoid unsupported claims;
- say that the answer is unknown when evidence is missing;
- preserve source references.

### 5.8 Generator or LLM

The LLM receives the augmented prompt and generates the final natural-language response. The application may attach citations by using the metadata of the retrieved documents.

---

## 6. RAG Data Flow

A RAG system has two major phases.

### 6.1 Indexing Phase

The indexing phase prepares a searchable knowledge base before users submit queries.

```text
Knowledge Sources
       ↓
Document Loading
       ↓
Parsing and Cleaning
       ↓
Chunking
       ↓
Embedding Generation
       ↓
Vectors + Text + Metadata
       ↓
Vector Store
```

This phase usually runs when a document is uploaded, changed, or re-indexed. The application should not normally load and embed every unchanged document again for each user query.

### 6.2 Retrieval and Generation Phase

This phase runs when a user submits a query.

```text
User Query
    ↓
Query Embedding
    ↓
Vector Similarity Search
    ↓
Relevant Chunks
    ↓
Prompt Augmentation
    ↓
LLM Generation
    ↓
Grounded Answer + Sources
```

### 6.3 Complete End-to-End Flow

```text
Documents → Load → Chunk → Embed → Vector Store
                                          ↑
User Query → Embed → Similarity Search → Retrieve
                                     ↓
                        Context + Query → LLM → Answer
```

---

## 7. Failures in RAG Systems

A RAG system can fail primarily at two stages: retrieval and generation.

### 7.1 Retriever Failure

A retriever failure occurs when the required information exists in the knowledge base, but the retriever does not return it.

Possible causes include:

- incomplete or poor-quality source documents;
- incorrect document parsing;
- unsuitable chunk size or overlap;
- a weak or mismatched embedding model;
- an ambiguous user query;
- an unsuitable retrieval strategy;
- an incorrect metadata filter;
- relevant documents being ranked too low.

```text
Poor Retrieval → Missing or Incorrect Context → Unreliable Answer
```

### 7.2 Generator Failure

A generator failure occurs when the correct context is retrieved, but the LLM still produces an incorrect, incomplete, or unsupported answer.

Possible causes include:

- weak prompt instructions;
- conflicting retrieved passages;
- excessive or noisy context;
- the LLM ignoring the supplied context;
- claims that are not supported by the retrieved documents;
- incorrect citation mapping.

```text
Correct Context + Weak Generation → Unreliable Answer
```

### 7.3 Failure Diagnosis

| Observation | Likely Failure | What to Inspect |
|---|---|---|
| The correct passage was not retrieved | Retriever failure | Chunks, embedding model, query, `k`, and filters |
| The correct passage was ranked too low | Ranking failure | Scores, hybrid search, and reranking |
| The correct context was present, but the answer was wrong | Generator failure | Prompt, model, and context order |
| The answer was correct, but the citation was wrong | Attribution failure | Metadata and source mapping |
| The answer came from an outdated document | Knowledge-base failure | Document versions and re-indexing |

### Golden debugging rule

> Inspect the retrieved documents first. Inspect the generated answer second.

A poor answer does not always mean that the LLM is the only problem. The retriever may have failed to provide the evidence required for a correct answer.

---

## 8. RAG vs Prompt Engineering vs Fine-Tuning

| Technique | What Changes? | Best Suited For |
|---|---|---|
| Prompt Engineering | Prompt instructions and structure | Clearer tasks and output control |
| RAG | External knowledge is added at runtime | Current, private, factual, and source-backed knowledge |
| Fine-Tuning | Model weights are updated through training | Behaviour, tone, format, or task adaptation |

### 8.1 RAG

```text
External Knowledge → Retrieval → Runtime Context → LLM
```

- Model parameters remain unchanged.
- Information can be updated by changing the knowledge base.
- Source attribution is possible.
- It is suitable for frequently changing factual knowledge.

### 8.2 Fine-Tuning

```text
Specialized Dataset → Training → Updated Model Parameters
```

- It requires a suitable training dataset and compute.
- It is useful for adapting behaviour, tone, format, or task performance.
- Updating frequently changing facts may require additional training.
- It is difficult to identify the exact training source behind a generated claim.

### 8.3 Choosing the Correct Technique

```text
Need updated company policies?           → RAG
Need answers from private documents?     → RAG
Need clearer instructions or formatting? → Prompt Engineering
Need a consistent specialized style?     → Fine-Tuning
Need knowledge and adapted behaviour?    → RAG + Fine-Tuning
```

RAG and fine-tuning are not mutually exclusive. A production system may use a fine-tuned model for specialized behaviour and RAG for current external knowledge.

---

## 9. Common Misunderstandings

### Does RAG train the LLM?

No. Standard RAG does not update the model's parameters. It provides retrieved information inside the prompt at runtime.

### Does RAG permanently add knowledge to the model?

No. The knowledge remains in an external store. The application retrieves the required information for each query.

### Does RAG completely stop hallucination?

No. RAG can reduce hallucination, but incorrect retrieval, conflicting context, or weak generation can still produce an unreliable answer.

### Is a vector database a complete RAG system?

No. A vector database is only one component. A complete RAG system also needs ingestion, retrieval, prompt augmentation, and generation.

### Are embeddings the final answer?

No. Embeddings are numerical representations used for semantic search. The LLM generates the final natural-language answer.

---

## 10. Quick Revision Sheet

```text
Major LLM limitations:
Knowledge cutoff + Hallucination + No exact sources + No private-data access

RAG solution:
Retrieve trusted external context at runtime and provide it to the LLM

Indexing phase:
Load → Parse → Chunk → Embed → Store

Query-time phase:
Query → Embed → Retrieve → Augment Prompt → Generate

Major failure categories:
Retriever Failure + Generator Failure

RAG vs Fine-Tuning:
External runtime knowledge vs updated model parameters
```

---

## 11. Interview Answer

> Retrieval-Augmented Generation, or RAG, is a technique that retrieves information relevant to a user's query from an external knowledge base and adds it to the LLM's context before generation. It helps address knowledge cutoffs, hallucinations, lack of source attribution, and the inability to access private data. A RAG pipeline includes document loading, chunking, embedding generation, vector storage, retrieval, prompt augmentation, and response generation. Its two major failure categories are retriever failure and generator failure.

---

## 12. Self-Test Questions

1. What is parametric knowledge?
2. Why should an LLM not be described as a conventional database of its training data?
3. What are the four major LLM limitations discussed in this chapter?
4. What does RAG stand for?
5. What do retrieval, augmentation, and generation mean?
6. Why should all available documents not be added directly to a prompt?
7. What information occupies an LLM's context window?
8. How are the responsibilities of a document loader and a text splitter different?
9. Why is an embedding model used in a RAG pipeline?
10. Why should documents and queries normally use the same embedding model?
11. Why does a vector store retain text and metadata in addition to vectors?
12. What is the responsibility of a retriever?
13. What is the difference between indexing and query-time processing?
14. What is the difference between retriever failure and generator failure?
15. Does RAG eliminate hallucination completely?
16. What is the main difference between RAG and fine-tuning?
17. When can RAG and fine-tuning be used together?

---

## Next Topic

The next topic is **Document Loaders**. A document loader reads content from a source such as a PDF, text file, website, or structured dataset and converts it into standardized document objects for the RAG indexing pipeline.
