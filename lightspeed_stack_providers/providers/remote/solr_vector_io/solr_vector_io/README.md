# Solr Vector IO Provider

A read-only vector_io provider implementation for llama-stack that integrates with Apache Solr's DenseVectorField and KNN search capabilities.

## Features

- **Read-only access** to Solr collections with vector embeddings
- **Vector similarity search** using Solr's KNN query parser
- **Keyword search** using Solr's text search (edismax)
- **Hybrid search** combining vector and keyword search with Solr's native reranking
- **Chunk window expansion** for retrieving extended context around matched chunks
- **Schema-agnostic** field mapping for flexible Solr schema support
- **OpenAI-compatible API** for vector store operations (read-only methods)

## Installation

Using uv:

```bash
# Sync "production" dependencies
uv sync

# Sync testing & optional dependencies (some tests will be skipped without the dependencies)
uv sync --all-extras
```

## Configuration

The provider requires the following configuration:

| Parameter             | Description                                                | Required   | Default                        | Example                        |
| -----------           | -------------                                              | ---------- | -----------------              | -----------------              |
| `solr_url`            | Base URL of the Solr server                                | Yes        | -                              | `"http://localhost:8983/solr"` |
| `collection_name`     | Name of the Solr collection to use                         | Yes        | -                              | `"portal"`                     |
| `vector_field`        | Name of the field containing `DenseVectorField` embeddings | Yes        | -                              | `"chunk_vector"`               |
| `content_field`       | Name of the field containing chunk text content            | Yes        | -                              | `"chunk"`                      |
| `id_field`            | Name of the field containing unique document identifier    | No         | `"id"`                         | `"chunk_id"`                   |
| `embedding_dimension` | Dimension of the embedding vectors                         | Yes        | -                              | `384`                          |
| `persistence`         | KV store configuration for metadata persistence            | No         | `None`                         | `None`                         |
| `request_timeout`     | Timeout for Solr requests in seconds                       | No         | `30`                           | `30`                           |
| `chunk_window_config` | Schema field mapping for chunk window expansion            | No         | `None`                         | `None`                         |

## Usage

This provider is designed to be used with llama-stack. The Solr collection should already exist and contain documents with:

1. A DenseVectorField for vector embeddings
2. A content/text field for the chunk text
3. Optional metadata fields

### Expected Solr Schema

#### Basic Schema (Required)

Documents should have fields compatible with the Chunk schema:
- **Content field**: Configured via `content_field` parameter (e.g., `chunk`, `content`, `text`)
- **Vector field**: DenseVectorField configured via `vector_field` parameter (e.g., `chunk_vector`)
- **ID field**: Configured via `id_field` parameter (defaults to `id`)
- Any additional fields will be mapped to chunk metadata

Example basic configuration:
```python
config = SolrVectorIOConfig(
    solr_url="http://localhost:8080/solr",
    collection_name="my_collection",
    vector_field="embedding",      # Your vector field name
    content_field="text",           # Your content field name
    embedding_dimension=384
)
```

#### Chunk Window Expansion Schema (Optional)

To enable chunk window expansion (context retrieval around matched chunks), configure `chunk_window_config` with these options to inform the Solr Vector IO provider how to find information in the Solr index.

**Chunk Document Fields:**

| Field                     | Description                          | Required   | OKP RAG Proto Example |
| -------                   | -------------                        | ---------- | ---------             |
| `chunk_parent_id_field`             | Chunk's parent document ID             | Yes  | `"parent_id"`              |
| `chunk_index_field`                 | Chunk's sequential position            | Yes  | `"chunk_index"`            |
| `chunk_token_count_field`           | Token count for the chunk              | Yes  | `"num_tokens"`             |
| `chunk_filter_query`                | Filter to identify chunk documents     | Yes  | `"is_chunk:true"`          |
| `chunk_expansion_boundary_fields`   | Fields that constrain window expansion | No   | `["parent_id", "heading"]` |

**Parent Document Fields:**

| Field                        | Description              | Required   | OKP RAG Proto Example |
| -------                      | -------------            | ---------- | ---------             |
| `parent_id_field`            | Solr document identifier | Yes        | `"id"`                |
| `parent_total_chunks_field`  | Total chunk count        | Yes        | `"total_chunks"`      |
| `parent_total_tokens_field`  | Total token count        | Yes        | `"total_tokens"`      |
| `parent_content_id_field`    | Content ID               | No         | `"doc_id"`            |
| `parent_content_title_field` | Content's title          | No         | `"title"`             |
| `parent_content_url_field`   | Content's URL            | No         | `"reference_url"`     |

#### Example RHOKP RAG Prototype configuration

Here's an example configuration with chunk windowing, and compatible with the Solr index from `images.paas.redhat.com/offline-kbase/okp-rag-proto:latest`:

```python
from src.solr_vector_io import SolrVectorIOConfig, ChunkWindowConfig

config = SolrVectorIOConfig(
    solr_url="http://localhost4:8080/solr",
    collection_name="portal",
    vector_field="chunk_vector",
    content_field="chunk",
    embedding_dimension=384,
    chunk_window_config=ChunkWindowConfig(
        chunk_parent_id_field="parent_id",
        chunk_index_field="chunk_index",
        chunk_token_count_field="num_tokens",
        parent_total_chunks_field="total_chunks",
        parent_total_tokens_field="total_tokens",
        parent_content_id_field="doc_id",
        parent_content_title_field="title",
        parent_content_url_field="reference_url",
        chunk_filter_query="is_chunk:true",
        chunk_expansion_boundary_fields=["parent_id"],
    )
)
```

**Note**: The `content_field` in the main config and `chunk_content_field` in the chunk window schema should typically be the same field.

## Chunk Window Expansion

When `chunk_window_config` is set, the provider automatically retrieves context around matched chunks by expanding to adjacent chunks within a token budget.

Configuration is set at initialization time via `ChunkWindowConfig`:

```python
config = SolrVectorIOConfig(
    solr_url="http://localhost:8080/solr",
    collection_name="my_collection",
    vector_field="chunk_vector",
    content_field="chunk",
    embedding_dimension=384,
    chunk_window_config=ChunkWindowConfig(
        # Required: Schema field mappings
        chunk_parent_id_field="parent_id",
        chunk_index_field="chunk_index",
        chunk_token_count_field="num_tokens",
        parent_total_chunks_field="total_chunks",
        parent_total_tokens_field="total_tokens",
        chunk_filter_query="is_chunk:true",

        # Optional: Expansion parameters (with defaults shown)
        token_budget=2048,        # Max tokens per context window
        min_chunk_gap=4,          # Min spacing between chunks from same doc
        min_chunk_window=4,       # Min chunks before windowing applies

        # Optional: Boundary fields constrain expansion to chunks sharing
        # the same values for these fields as the matched chunk.
        # chunk_parent_id_field is always used as a boundary regardless
        # of this setting. Default is None (no additional boundaries).
        chunk_expansion_boundary_fields=["parent_id", "heading"],
    )
)

# Queries automatically use chunk window expansion
response = await adapter.query_chunks(
    vector_db_id="my-store",
    query="How to fix security issues?",
    params={
        "k": 5,
        "score_threshold": 0.0
    }
)
```

This feature:
- Automatically expands matched chunks to include adjacent chunks
- Finds the best matching chunk via vector/keyword/hybrid search
- Expands bidirectionally within token budget limits
- Prevents duplicate context from the same document
- Returns concatenated chunk content as a single result
- Optionally constrains expansion to chunks sharing the same values for configured boundary fields (e.g. heading/section)

## Hybrid Search

Hybrid search uses Solr's native query reranking capabilities. You can configure the boost parameters:

```python
params = {
    "reranker_params": {
        "vector_boost": 1.0,  # Weight for vector similarity
        "keyword_boost": 1.0,  # Weight for text search
    }
}
```

## Limitations

This is a **read-only** provider. The following operations are not supported:
- `insert_chunks` - raises NotImplementedError
- `delete_chunks` - raises NotImplementedError
- OpenAI vector store creation/modification methods

Use Solr's native APIs or tools to manage the index and documents.

## Development

### Running Tests

#### Initial setup for tests

The tests depend on the OKP Solr RAG prototype container image.

1. Pull the Offline Knowledge Portal with public RAG prototype image:

```
podman pull images.paas.redhat.com/offline-kbase/okp-rag-proto:nov20
```

**NOTE:** this requires a VPN connection and is a temporary image for testing.

2. Run the OKP image:

```
podman run --rm -p 8080:8080 -d --name okp okp-rag-proto:latest
```

3. Verify OKP image is running locally by opening <http://127.0.0.1:8080/> in your browser, or by running `podman logs okp -f` and waiting for the `Happy searching!` log message from Solr.

#### Running tests

Once the OKP Solr RAG prototype is running, the tests can be run against it.

```bash
# Run with pytest directly (all tests)
uv run tests.py

# Skip slow tests (the ones that produce real embeddings for test purposes)
uv run tests.py -m "not slow"

# Run specific test suites
uv run tests.py -m basic
uv run tests.py -m search
uv run tests.py -m chunk_window
uv run tests.py -m persistence

# Run tests with debug logging
uv run tests.py --log-cli-level=DEBUG -v

# Run a specific test (where "pattern" is a string occurring in the test(s) name)
uv run tests.py -k pattern
```

Test markers:
- `basic` - Basic functionality (adapter init, registration, etc.)
- `search` - Search functionality (vector, keyword, hybrid)
- `chunk_window` - Chunk window expansion tests
- `persistence` - Persistence/KV store tests
- `embeddings` - Tests using real embedding models
- `slow` - Slow tests 

## License

TBD
