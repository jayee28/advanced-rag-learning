# Vector Stores and Similarity Search in RAG

## 1. What Is a Vector Store?

A vector store stores embedding vectors and supports searching for similar vectors.

In a RAG application, vectors are typically associated with:

- A unique ID
- The original text chunk
- Metadata, such as source filename and page number

Depending on the implementation, text and metadata may be stored alongside the vectors or in a separate linked store.

| ID | Text Chunk | Embedding | Metadata |
|---|---|---|---|
| chunk-1 | RAG combines retrieval and generation. | [0.2, 0.5, ...] | page: 1 |
| chunk-2 | Embeddings represent text numerically. | [0.7, 0.1, ...] | page: 2 |

The numbers above are illustrative.

**Key takeaway:** A vector store helps retrieve relevant content. The LLM generates the final answer.

## 2. Where Does a Vector Store Fit in RAG?

### Indexing Phase

1. Load source documents.
2. Extract their text.
3. Split the text into chunks.
4. Generate an embedding for each chunk.
5. Store the vectors and their associated content.
6. Build or update the search index.

### Query Phase

1. Receive the user's question.
2. Generate its embedding.
3. Search for similar document vectors.
4. Retrieve the associated chunks.
5. Pass the question and retrieved context to the LLM.
6. Generate an answer.

Document embeddings are generally created during ingestion and reused across queries.

## 3. Embedding Space and Dimensions

An embedding represents text as a point in a multidimensional space.

- A 2D point needs two coordinates: `(x, y)`.
- A 3D point needs three coordinates: `(x, y, z)`.
- A 768-dimensional embedding needs 768 coordinates.

The embedding model and its configuration determine the output dimension.

A short query and a long document chunk can both produce 768-dimensional vectors. Their values differ, but their number of coordinates remains the same.

### Why Must Query and Document Dimensions Match?

Similarity calculations compare corresponding coordinates.

For example:

Query:    [q1, q2, q3]
Document: [d1, d2, d3]

Both vectors must belong to a compatible embedding space.

**Matching dimensions alone is insufficient.** Use the same embedding model and compatible settings for documents and queries, including any model-specific input formatting.

Different models can produce vectors of the same size without producing compatible representations.

### Dimension vs Magnitude

Dimension is the number of coordinates.

Magnitude is the vector's length from the origin.

For the vector:

v = [3, 4]

- Dimension = 2
- Magnitude = √(3² + 4²) = 5

Normalizing a nonzero vector to unit length changes its magnitude to 1. It does not change its dimension.

## 4. Document and Query Embedding Shapes

For `N` document chunks and embedding dimension `d`:

| Output | Structure | Conceptual Shape |
|---|---|---|
| One query embedding | List of floats | `(d,)` |
| Document embeddings | List of lists of floats | `(N, d)` |
| A batch containing one document | List containing one vector | `(1, d)` |

A query vector is a 1D list, but it still represents a point in a `d`-dimensional embedding space.

**Array dimensionality and embedding dimensionality are different concepts.**

## 5. Similarity Search

Similarity search finds stored vectors that are close to a query vector according to a chosen metric.

Common measures include:

| Measure | Interpretation |
|---|---|
| Cosine similarity | Compares vector direction; higher is more similar |
| Dot product | Depends on direction and magnitude; higher ranks first |
| Euclidean distance | Measures straight-line distance; lower is closer |

For unit-normalized vectors, cosine similarity and dot product are equivalent. Euclidean distance also gives an equivalent nearest-neighbor ranking in that case.

Semantic similarity is an approximation learned by the embedding model. A nearby vector does not guarantee that its text answers the question.

### What Does Top-k Mean?

`k` is the number of results requested.

For example:

- `k = 3`: request the three nearest results.
- `k = 5`: request the five nearest results.

Top-k does not require a minimum relevance score. Even the closest available chunks may be irrelevant.

## 6. Brute-Force Search

Brute-force search compares the query vector against every stored vector.

### Steps

1. Generate the query embedding.
2. Calculate its similarity or distance to every stored vector.
3. Select the best `k` results.
4. Return their associated content.

### Example

Suppose the store contains 10,000 vectors and `k = 3`.

Brute force still compares the query against all 10,000 vectors before selecting the best three.

**`k` controls the result count, not the number of comparisons.**

### Computational Cost

For:

- `N`: number of stored vectors
- `d`: dimensions per vector

Calculating all similarity scores typically costs:

O(N × d)

Top-k selection adds additional work. A full sort is one option, but it is not required.

### Advantages

- Simple to understand and implement
- Produces exact nearest neighbors under the chosen metric
- Useful as a baseline for evaluating approximate search
- Can be practical for small datasets

### Limitations

- Work increases with the number of vectors
- Higher dimensions increase comparison cost
- Repeated searches over large datasets can be expensive

## 7. Exact Search

Exact search returns the actual nearest neighbors according to the selected metric.

Brute force is an exact-search method, but not every exact-search method must use brute force.

### What Does “Exact” Guarantee?

It guarantees the nearest vectors under the metric.

It does not guarantee:

- That the extracted document text is correct
- That the embedding captures every relevant detail
- That the retrieved chunk answers the question
- That the LLM will generate a correct answer

**Exact vector search is not the same as guaranteed semantic correctness.**

## 8. Why Indexing Is Needed

An index organizes data into a structure that supports search.

Vector indexes can reduce search work by guiding the search toward promising candidates.

The index adds costs of its own:

- Construction time
- Storage or memory
- Maintenance when data changes

Not every index is approximate. For example, a flat index can still perform exhaustive exact search.

## 9. Approximate Nearest Neighbor Search

Approximate Nearest Neighbor search, or ANN, aims to find close neighbors efficiently without guaranteeing the exact top-k results.

ANN methods commonly avoid comparing the query against every stored vector.

### Main Trade-off

Searching more candidates usually improves recall but requires more work.

Searching fewer candidates is often faster but may miss true nearest neighbors.

Actual performance depends on the dataset, index, hardware, and configuration.

### ANN Recall@k

To measure how well ANN matches exact search:

Recall@k =
    Number of exact top-k neighbors found by ANN
    ÷ k

Example:

Exact top-5: A, B, C, D, E
ANN top-5:   A, B, C, F, G

Recall@5 = 3 / 5 = 0.6 = 60%

This measures agreement with exact nearest-neighbor results.

It is different from measuring whether retrieved chunks are relevant to the user's actual question.

## 10. IVF: Clustering-Based Search

IVF stands for **Inverted File Index**.

A common IVF approach partitions the vector space into clusters and associates vectors with those clusters.

### Index Construction

1. Learn representative cluster centers, often using k-means.
2. Assign each document vector to a cluster.
3. Store the vectors or their representations in that cluster's list.

### Query Search

1. Compare the query with cluster centers.
2. Select nearby clusters.
3. Search the vectors inside those selected clusters.
4. Return the best candidates.

### Example

Suppose 10,000 vectors are distributed across 100 clusters.

If search examines 5 clusters, it searches only their candidate vectors rather than all 10,000 document vectors.

If clusters were evenly sized, those 5 clusters would contain about 500 vectors. Real clusters may have unequal sizes, and selecting clusters also involves work.

### Important Parameters

| Parameter | Purpose |
|---|---|
| `nlist` | Number of clusters or inverted lists |
| `nprobe` | Number of lists searched for a query |

Increasing `nprobe` generally improves recall while increasing search work.

### Why Can IVF Miss a Neighbor?

A relevant vector may belong to a cluster that the search does not examine.

Cluster boundaries do not guarantee that every nearest neighbor lies in the closest cluster.

**Key takeaway:** IVF narrows the search to selected regions of the vector space.

## 11. HNSW: Graph-Based Search

HNSW stands for **Hierarchical Navigable Small World**.

It organizes vectors into a graph with multiple layers.

- Nodes represent vectors.
- Edges connect selected neighboring nodes.
- Upper layers contain fewer nodes.
- The bottom layer contains all indexed vectors.

### Search Intuition

Upper layers help move quickly toward the query's neighborhood.

Lower layers allow a more detailed search around promising candidates.

### Simplified Search Process

1. Start at an entry point in an upper layer.
2. Follow connections toward nodes closer to the query.
3. Move down through the layers.
4. Explore candidate neighbors in the bottom layer.
5. Return the best results found.

At the bottom layer, search can maintain multiple candidates rather than following only one path.

### Important Parameters

| Parameter | Purpose |
|---|---|
| `M` | Controls graph connectivity |
| `efConstruction` | Controls candidate exploration while building the graph |
| `efSearch` | Controls candidate exploration during querying |

Parameter names can vary by implementation.

Higher connectivity and wider exploration generally require more resources and can improve recall.

### Why Can HNSW Miss a Neighbor?

The search explores a limited portion of the graph. It may stop without visiting a true nearest neighbor.

**Key takeaway:** HNSW uses graph connections and a hierarchy to guide search toward nearby vectors.

## 12. Exact Search vs IVF vs HNSW

| Aspect | Brute Force | IVF | HNSW |
|---|---|---|---|
| Main structure | Flat collection of vectors | Clusters and inverted lists | Layered graph |
| Query behavior | Compare every vector | Search selected clusters | Explore graph neighbors |
| Exact top-k guaranteed | Yes, under the chosen metric | Generally no when only some clusters are searched | No |
| Main search control | Result count `k` | `nprobe` | `efSearch` |
| Extra index preparation | Minimal structural preparation | Learn clusters and assign vectors | Build graph connections |
| Main trade-off | Query cost grows with dataset size | Search coverage versus speed | Search coverage, speed, and memory |

IVF and HNSW are two important ANN approaches, not the only ones.

## 13. Vector Store vs Vector Database

The terms are often used loosely.

A vector store is a broad abstraction for storing and searching vectors.

A vector database typically includes broader data-management capabilities, such as:

- Persistent storage
- Metadata filtering
- Updates and deletions
- Concurrent access
- Operational management

Capabilities vary by product and configuration. Check the implementation rather than relying only on the label.

## 14. Retrieval Quality in RAG

Fast search is useful only if the retrieved context helps answer the question.

Retrieval quality depends on:

- Text extraction quality
- Chunk boundaries and overlap
- Embedding model suitability
- Query formulation
- Search metric and index settings
- Metadata filters
- Number of retrieved chunks

An ANN index may find approximate neighbors very efficiently, but it cannot recover information that was never extracted or indexed.

## 15. Evaluation and Debugging

Evaluate retrieval and generation separately.

### Retrieval Evaluation

Ask:

- Was the required information indexed?
- Did the relevant chunks appear in the results?
- Were useful chunks ranked near the top?

### Generation Evaluation

Ask:

- Is the answer correct?
- Does it address the question?
- Are its claims supported by the retrieved context?
- Are citations accurate?

### Example

Source document:

“Refunds are allowed within 7 days.”

| Observation | Possible Area to Investigate |
|---|---|
| Refund text is missing from extracted content | Parsing or ingestion |
| Text exists but is split without necessary context | Chunking |
| Correct chunk is indexed but not retrieved | Retrieval |
| Correct chunk is retrieved but answer says 30 days | Generation |
| Answer says 7 days but cites the wrong page | Citation handling |

These are starting points for investigation, not automatic diagnoses.

### A Simple Evaluation Dataset

Create 10–15 questions from the PDF and record:

| Question | Expected Answer | Source Page | Retrieved Correct Chunk? | Answer Correct? |
|---|---|---|---|---|
| What is the refund period? | 7 days | 4 | Yes / No | Yes / No |

Use the same questions when comparing chunking settings or search configurations.

### Correctness vs Faithfulness

- **Correctness:** Does the answer match the expected facts?
- **Faithfulness:** Are its claims supported by the supplied context?

An answer can follow an incorrect source faithfully and still be factually wrong.

## Key Takeaways

- A vector store supports similarity-based retrieval.
- Query and document vectors need compatible embedding spaces.
- Brute force compares the query against every vector.
- Exact search guarantees nearest vectors under a metric, not correct answers.
- Indexing organizes vectors for search.
- ANN trades exact-neighbor guarantees for search efficiency.
- IVF searches selected clusters.
- HNSW navigates a layered graph.
- Search quality and search speed should be evaluated together.
- RAG debugging requires inspecting each pipeline stage.

## Quick Revision

| Term | Meaning |
|---|---|
| Embedding | Numerical representation of an input |
| Dimension | Number of coordinates in a vector |
| Vector store | System or abstraction for storing and searching vectors |
| Top-k | Number of nearest results requested |
| Brute force | Compare the query with every stored vector |
| Exact search | Return the actual nearest neighbors under a metric |
| Index | Data structure used to support search |
| ANN | Approximate Nearest Neighbor search |
| IVF | Index that searches selected inverted lists, commonly organized by clusters |
| HNSW | Hierarchical graph-based ANN method |
| Recall@k for ANN | Fraction of exact top-k neighbors recovered |
| Faithfulness | Support for generated claims in the supplied context |

## Interview Questions

1. What role does a vector store play in RAG?
2. Why must query and document embeddings use compatible models?
3. How does embedding dimension differ from array dimensionality?
4. Does `k = 3` reduce brute-force search to three comparisons?
5. What is the similarity-computation cost of brute-force search?
6. Why does exact search not guarantee a correct RAG answer?
7. What does a vector index do?
8. What trade-off does ANN introduce?
9. How does IVF narrow the search?
10. What happens when `nprobe` increases?
11. How does HNSW use its hierarchy?
12. What does `efSearch` control?
13. How would you measure ANN recall against exact search?
14. How would you distinguish a retrieval failure from a generation failure?
15. Why should latency and retrieval quality be measured together?