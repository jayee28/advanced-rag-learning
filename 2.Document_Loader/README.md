# Document Loaders for RAG — Complete Notes

These notes cover the document loaders practised in this module:

1. JSON documents
2. Web pages
3. Recursive website crawling
4. Text files
5. PDF documents with `pypdf`, `pdfplumber`, and `pdfminer.six`
6. Normal loading and lazy loading

## 1. What Is a Document Loader?

A document loader reads data from a knowledge source and converts it into a standard format that a RAG application can process.

Different sources store information differently:

- JSON contains keys, values, arrays, and nested objects.
- Web pages contain HTML, navigation, scripts, and links.
- Text files contain unstructured plain text.
- PDFs store text using page layout and coordinates and may also contain images.

A loader hides these source-specific differences and returns content together with metadata.

In LangChain, the standard output is a `Document` object:

```python
Document(
    page_content="The extracted text",
    metadata={"source": "path-or-url"},
)
```

## 2. Why Are Document Loaders Required?

An LLM or embedding model cannot directly understand every file format. Before the information can be chunked, embedded, stored, and retrieved, it must be extracted and normalized.

Document loaders solve the following problems:

- Reading different file and source formats
- Extracting useful text
- Preserving metadata such as source, page number, and URL
- Providing a consistent output structure
- Supporting batch and incremental processing
- Separating data ingestion from the rest of the RAG system

## 3. Document Loading in a RAG Pipeline

```text
Knowledge source
      |
      v
Document loader
      |
      v
Documents: page_content + metadata
      |
      v
Cleaning -> splitting -> embeddings -> vector store -> retrieval
```

A document loader does not perform chunking, embedding, vector storage, or retrieval. Its responsibility is loading and normalizing source data.

## 4. Common Loader Interface

Most LangChain document loaders provide two methods.

### `load()`

Loads all documents and returns a list:

```python
docs = loader.load()

print(type(docs))
print(len(docs))
```

Use it when the source is small enough to fit comfortably in memory.

### `lazy_load()`

Returns an iterator and produces documents one at a time:

```python
docs = loader.lazy_load()

for doc in docs:
    print(doc.page_content)
```

Use it for large files, many files, or website crawling when incremental processing is useful.

Important: a generator is normally consumed after one complete iteration.

```python
docs = loader.lazy_load()

for doc in docs:
    print(doc)

# The same generator may now be exhausted.
```

Converting it to a list makes it reusable but eventually keeps every document in memory:

```python
docs = list(loader.lazy_load())
```

## 5. Installation with uv

For the older LangChain loader examples:

```bash
uv add langchain-core langchain-community
```

Additional parser dependencies:

```bash
uv add jq beautifulsoup4 pypdf pdfplumber pdfminer-six
```

`langchain-community` is being sunset and is no longer actively maintained. It can still be used to understand existing course examples, but new applications should prefer maintained standalone integrations or small application-level loaders built directly with the parser libraries.

## 6. JSONLoader

JSON is structured data. A JSON loader needs to know which part of the structure should become one document. `jq_schema` describes that selection.

Example JSON:

```json
{
  "products": [
    {
      "productID": "0000001",
      "manufacturer": "Zara",
      "productName": "PINSTRIPE COAT",
      "Description": "Oversize-fit coat made of a viscose blend fabric.",
      "price": 4900,
      "category": "Men Clothes"
    },
    {
      "productID": "0000002",
      "manufacturer": "Zara",
      "productName": "TECHNICAL TRENCH COAT",
      "Description": "Trench coat made of technical fabric.",
      "price": 4900,
      "category": "Men Clothes"
    }
  ]
}
```

### Loading each complete product object

```python
from langchain_community.document_loaders import JSONLoader

loader = JSONLoader(
    file_path="./knowledge-source/apparels.json",
    jq_schema=".products[]",
    text_content=False,
)

docs = loader.load()

for doc in docs:
    print(doc.page_content)
    print(doc.metadata)
```

Here:

- `.` means the root JSON object.
- `.products` selects the `products` property.
- `[]` iterates through every element in the array.
- `.products[]` therefore selects every complete product object.
- `text_content=False` allows the selected value to be a JSON object instead of requiring a string.

### Loading only the description field

```python
loader = JSONLoader(
    file_path="./knowledge-source/apparels.json",
    jq_schema=".products[].Description",
    text_content=True,
)
```

### Common JSONLoader mistake

This schema is wrong for the example above:

```python
jq_schema=".data[].text"
```

The JSON contains `products`, not `data`, and its objects do not contain a lowercase `text` field. The schema must match the actual JSON structure and capitalization.

## 7. WebBaseLoader

`WebBaseLoader` loads the web page or pages explicitly supplied to it.

```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    web_paths=("https://example.com",)
)

docs = loader.load()

for doc in docs:
    print(doc.page_content[:500])
    print(doc.metadata)
```

Use it when you already know the exact URLs that must be loaded.

### Loading selected HTML elements

```python
from bs4 import SoupStrainer
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    web_paths=("https://example.com",),
    bs_kwargs={
        "parse_only": SoupStrainer(["main", "article"]),
    },
)

docs = loader.load()
```

Filtering page elements can reduce noise from menus, navigation, sidebars, and footers.

## 8. RecursiveUrlLoader

`RecursiveUrlLoader` begins with one URL, extracts links, and recursively visits related pages.

```python
from bs4 import BeautifulSoup
from langchain_community.document_loaders import RecursiveUrlLoader


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body

    if main is None:
        return ""

    return main.get_text(separator="\n", strip=True)


loader = RecursiveUrlLoader(
    url="https://docs.python.org/3/tutorial/",
    max_depth=2,
    extractor=extract_text,
    prevent_outside=True,
    check_response_status=True,
)

for doc in loader.lazy_load():
    print(doc.metadata.get("source"))
    print(doc.page_content[:500])
```

Important parameters:

| Parameter | Purpose |
| --- | --- |
| `url` | Starting URL |
| `max_depth` | Maximum recursive link depth |
| `extractor` | Converts raw HTML into useful text |
| `prevent_outside=True` | Prevents crawling outside the starting domain |
| `exclude_dirs` | Excludes selected paths |
| `timeout` | Limits request waiting time |
| `check_response_status=True` | Raises an error for unsuccessful HTTP responses |

Start with `max_depth=1` or `2`. A high depth can cause unnecessary requests and collect duplicate, irrelevant, or extremely large amounts of content.

### WebBaseLoader vs RecursiveUrlLoader

| Feature | WebBaseLoader | RecursiveUrlLoader |
| --- | --- | --- |
| Input | One or more explicit URLs | One starting URL |
| Follows links | No | Yes |
| Best use | Selected pages | Documentation sites or connected pages |
| Main risk | Noisy HTML | Large and uncontrolled crawl |

Before crawling a website, respect its access rules, terms, rate limits, and `robots.txt` guidance.

## 9. TextLoader

`TextLoader` reads a plain-text file and normally returns one `Document` for the complete file.

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader(
    file_path="./knowledge-source/sample.txt",
    encoding="utf-8",
)

docs = loader.load()

print("Total documents:", len(docs))
print(docs[0].page_content)
print(docs[0].metadata)
```

Typical metadata:

```python
{"source": "./knowledge-source/sample.txt"}
```

For a large text file, load it first and then use a text splitter. Loading and splitting are separate steps.

## 10. PDF Loading

The three parser libraries practised are:

- `pypdf`
- `pdfplumber`
- `pdfminer.six`

### pypdf

`pypdf` is a lightweight default for regular, machine-generated PDFs.

```python
from pypdf import PdfReader

reader = PdfReader("./knowledge-source/sample.pdf")

print("Total pages:", len(reader.pages))

for page_number, page in enumerate(reader.pages):
    text = page.extract_text() or ""
    print(f"\n--- Page {page_number + 1} ---")
    print(text[:500])
```

### Creating LangChain Documents with pypdf

```python
from collections.abc import Iterator
from pypdf import PdfReader
from langchain_core.documents import Document


def lazy_load_pdf(file_path: str) -> Iterator[Document]:
    reader = PdfReader(file_path)

    for page_number, page in enumerate(reader.pages):
        yield Document(
            page_content=page.extract_text() or "",
            metadata={
                "source": file_path,
                "page": page_number,
                "loader": "pypdf",
            },
        )


docs = list(lazy_load_pdf("./knowledge-source/sample.pdf"))
```

### pdfplumber

`pdfplumber` is useful when layout, coordinates, lines, or tables matter.

```python
import pdfplumber

with pdfplumber.open("./knowledge-source/sample.pdf") as pdf:
    for page_number, page in enumerate(pdf.pages):
        text = page.extract_text() or ""
        tables = page.extract_tables()

        print(f"Page: {page_number + 1}")
        print(text[:500])
        print("Tables:", len(tables))
```

### pdfminer.six

`pdfminer.six` provides lower-level text and layout analysis.

```python
from pdfminer.high_level import extract_text

text = extract_text("./knowledge-source/sample.pdf")
print(text[:1000])
```

The high-level `extract_text()` function normally returns the text of the complete PDF. More custom processing is required when separate page-level `Document` objects and metadata are needed.

### PDF parser comparison

| Parser | Best suited for | Limitation |
| --- | --- | --- |
| `pypdf` | Normal text-based PDFs and quick RAG prototypes | Complex layouts and tables may be inaccurate |
| `pdfplumber` | Tables and detailed page inspection | Detailed processing can be slower |
| `pdfminer.six` | Lower-level text and layout analysis | Needs more custom code for convenient page-level Documents |

Recommended workflow:

1. Start with `pypdf`.
2. Inspect the extracted content.
3. Try `pdfplumber` when tables or layout are important.
4. Use `pdfminer.six` when detailed extraction control is required.

## 11. Legacy LangChain PDF Loaders

Older examples may use:

```python
from langchain_community.document_loaders import (
    PyPDFLoader,
    PDFPlumberLoader,
    PDFMinerLoader,
)
```

Example:

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("./knowledge-source/sample.pdf")
docs = loader.load()

for doc in docs:
    print(doc.page_content[:500])
    print(doc.metadata)
```

The deprecation warning does not mean that the current code immediately stops working. It means that the package is no longer the preferred actively maintained location. Use it to understand an existing course, but choose maintained standalone integrations or direct parser libraries for new production code.

## 12. Scanned PDFs and OCR

The three PDF parsers primarily read an existing text layer. A scanned PDF may contain only images, causing empty or nearly empty extracted content.

```python
if not docs[0].page_content.strip():
    print("No text layer detected; OCR may be required.")
```

For scanned PDFs, use an OCR-capable tool such as Tesseract, Docling, or a document-intelligence service.

## 13. Loading vs Splitting

A loader may produce one large document or one document per page. A text splitter creates smaller chunks suitable for retrieval.

```bash
uv add langchain-text-splitters
```

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
)

chunks = splitter.split_documents(docs)

print("Loaded documents:", len(docs))
print("Generated chunks:", len(chunks))
```

```text
Loader: source -> Document objects
Splitter: Document objects -> smaller Document chunks
```

## 14. Common Problems and Solutions

| Problem | Likely reason | Solution |
| --- | --- | --- |
| JSON loader returns nothing | Incorrect `jq_schema` | Match the real keys, nesting, arrays, and capitalization |
| Web page contains menus and scripts | Entire HTML was extracted | Use Beautiful Soup filtering or an extractor function |
| Recursive crawl loads too many pages | Depth is too high | Reduce `max_depth` and exclude irrelevant paths |
| Text file shows decoding errors | Wrong encoding | Specify the correct encoding, commonly UTF-8 |
| PDF text is empty | Scanned/image-only PDF | Use OCR |
| PDF reading order is incorrect | Columns or complex layout | Try `pdfplumber` or a layout-aware parser |
| PDF tables are broken | Positioned text is not a real table | Use table extraction or a specialist parser |
| Memory use is high | Everything was loaded at once | Process with `lazy_load()` or a custom generator |
| Source information is lost | Metadata was discarded | Preserve source URL/path and page number |

## 15. Good Practices for RAG Ingestion

- Validate whether any useful content was extracted.
- Preserve source, page, URL, title, and other useful metadata.
- Remove duplicated navigation, headers, and footers.
- Use lazy loading for large sources.
- Do not embed empty documents.
- Inspect sample outputs before indexing everything.
- Keep loader logic separate from splitting and embedding logic.
- Respect website permissions and rate limits.
- Never commit passwords, API keys, or sensitive source data.
- Evaluate extraction quality with real documents from the target domain.

## 16. Final Revision

```text
JSONLoader
  -> Selects JSON objects or fields using jq_schema

WebBaseLoader
  -> Loads explicitly supplied web pages

RecursiveUrlLoader
  -> Starts from one URL and follows related links

TextLoader
  -> Loads plain-text files

pypdf
  -> Good default for regular PDFs

pdfplumber
  -> Useful for tables and page layout

pdfminer.six
  -> Detailed, lower-level PDF text analysis
```

The most important idea is that every knowledge source has a different structure, but the remaining RAG pipeline needs a consistent representation. Document loaders create that bridge.
