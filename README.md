# GraphRAG Fraud / SMS Spam Experiment

A local GraphRAG experiment that turns a CSV of SMS spam messages into a queryable knowledge graph. The project uses **Groq** for LLM-based graph extraction and a local **SentenceTransformers** embeddings API, avoiding paid OpenAI API usage.
To begin with current the project only uses a pilot set of SMS spam messages. When there is possibility of large compute, more messages can be added to the input dataset.

## What this project does

Given an input CSV containing SMS messages, GraphRAG:

1. Loads the SMS data and divides it into text units.
2. Uses a Groq-hosted language model to extract entities and relationships.
3. Builds a knowledge graph from those extracted entities and relationships.
4. Groups related entities into communities and produces community reports.
5. Generates local vector embeddings for retrieval.
6. Lets you run local and global natural-language queries over the resulting graph.

For example, an SMS such as:

> Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...

can yield entities such as `Jurong Point`, `Bugis`, `Great World`, and `Cine`, alongside relationships based on their co-occurrence and descriptions in the SMS data.

## Architecture

```text
SMS spam CSV
    |
    v
GraphRAG indexing pipeline
    |---------------------> Groq LLM
    |                         - entity extraction
    |                         - relationship extraction
    |                         - community reports
    |
    |---------------------> Local embeddings server
                              - all-MiniLM-L6-v2
                              - OpenAI-compatible /v1/embeddings API
    |
    v
GraphRAG output artifacts
    - entities.parquet
    - relationships.parquet
    - text_units.parquet
    - community_reports.parquet
    - documents.parquet
    |
    v
GraphRAG local/global queries
```

## Requirements

- Python 3.11+
- A Groq API key
- A working GraphRAG installation
- `sentence-transformers`, `torch`, `fastapi`, `uvicorn`, and `pydantic`
- An SMS spam CSV stored in the project input directory

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install graphrag sentence-transformers fastapi uvicorn pydantic torch
```

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

Do not commit `.env` or API keys to version control.

## Local embeddings server

Groq provides LLM inference but is not being used here for embeddings. This project runs embeddings locally with the `all-MiniLM-L6-v2` SentenceTransformers model.

Create `local_embeddings_server.py` in the project root:

```python
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Explicit CPU mode avoids Apple Metal/MPS command-buffer crashes.
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

class EmbeddingsRequest(BaseModel):
    model: str
    input: list[str]

@app.post("/v1/embeddings")
def embeddings(req: EmbeddingsRequest):
    vecs = model.encode(req.input, convert_to_numpy=True).tolist()
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": i,
                "embedding": vector,
            }
            for i, vector in enumerate(vecs)
        ],
        "model": req.model or "all-MiniLM-L6-v2",
        "usage": {
            "prompt_tokens": 0,
            "total_tokens": 0,
        },
    }
```

Start it in a separate terminal and leave it running while indexing or querying:

```bash
source .venv/bin/activate
python -m uvicorn local_embeddings_server:app --host 127.0.0.1 --port 8686
```

### Verify the embeddings endpoint

In another terminal:

```bash
curl -v http://127.0.0.1:8686/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummy-key" \
  -d '{"input": ["test"], "model": "text-embedding-3-small"}'
```

A successful setup returns HTTP `200 OK` and JSON with a `data` list containing an `embedding` vector.

> The endpoint expects `input` to be a list. Sending `"input": "test"` produces an HTTP 422 validation error.

## GraphRAG configuration

Configure GraphRAG so that completions use Groq and embeddings use the local server. In `settings.yaml`, use the equivalent of:

```yaml
completion_models:
  default_completion_model:
    model_provider: groq
    model: openai/gpt-oss-20b
    auth_method: api_key
    api_key: ${GROQ_API_KEY}
    retry:
      type: exponential_backoff

embedding_models:
  default_embedding_model:
    model_provider: openai
    model: all-MiniLM-L6-v2
    auth_method: api_key
    api_key: dummy-key
    api_base: http://127.0.0.1:8686/v1
```

### Important model naming note

With LiteLLM, `groq/` identifies the provider. A Groq completion model should look like:

```yaml
model: groq/llama-3.1-8b-instant
```

Do not put `GROQ_API_KEY` in an OpenAI cloud configuration. The embedding provider is named `openai` only because the local server implements an OpenAI-compatible API schema; it does **not** send requests to OpenAI because `api_base` points to localhost.

If you use a different Groq model, first verify that the model is available to your Groq account. A `model_not_found` error means that the requested Groq model name is unavailable or not enabled for the account.

## Input data

Place the SMS CSV in GraphRAG's configured input directory, commonly:

```text
input/
  sms_spam.csv
```

GraphRAG supports CSV input. The content should have a clear message/text column. Useful columns include:

```text
label,message
spam,"Congratulations, claim your prize now..."
ham,"See you at 7 PM"
```

For fraud/smishing analysis, include messages containing concepts relevant to the intended threat model, such as:

- Bank or financial institution names
- OTPs, account verification, or account alerts
- Urgent transfer or "safe account" requests
- Suspicious links and shortened URLs
- Prize, delivery, loan, tax, or payment scams

A small sample of 38 rows is enough to validate the pipeline, but it may not contain enough examples of every theme. Consequently, some queries may be well supported while others correctly return that the information is unavailable.

## Build the index

First validate the configuration without executing workflows:

```bash
python -m graphrag index --dry-run
```

A silent return with no error is normal: dry-run validates the configuration and exits without building the index.

Run the real indexing pipeline:

```bash
python -m graphrag index
```

A successful run ends with:

```text
Pipeline complete
```

Typical successful workflows include:

```text
load_input_documents
create_base_text_units
create_final_documents
extract_graph
finalize_graph
extract_covariates
create_communities
create_final_text_units
create_community_reports
generate_text_embeddings
```

## Query the graph

Keep the local embeddings server running, then query from the project root.

### Global search

Use global search for corpus-wide themes and patterns:

```bash
python -m graphrag query \
  --root . \
  --method global \
  "What are the main themes in these SMS messages?"
```

Example questions:

```bash
python -m graphrag query --root . --method global \
  "What kinds of promotional tactics occur across the SMS spam dataset?"

python -m graphrag query --root . --method global \
  "What recurring persuasion tactics are used in the messages?"
```

### Local search

Use local search for specific entities, locations, phrases, or message patterns:

```bash
python -m graphrag query \
  --root . \
  --method local \
  "Show me an example of an SMS spam message that contains location names."
```

Example questions:

```bash
python -m graphrag query --root . --method local \
  "Which messages mention Jurong Point, Bugis, or Great World?"

python -m graphrag query --root . --method local \
  "Show messages that use urgency or a limited-time offer."
```

If the query asks about a concept absent or rare in the 38-row input, such as bank messages or OTP scams, GraphRAG may abstain. This is preferable to inventing an unsupported answer.

## Understanding the Parquet output

GraphRAG stores its artifacts in the configured output directory, often resembling:

```text
output/<timestamp>/artifacts/
```

You can inspect `.parquet` files with a local Python script, a desktop data tool, or an online Parquet viewer. Do not upload sensitive data to public online tools.

### `entities.parquet`

Each row is an **entity node** in the knowledge graph: a person, organization, location, product, concept, or other item that the LLM judged meaningful in the text.

Useful fields commonly include:

| Field | Meaning |
|---|---|
| `title` | The extracted entity name, such as `Jurong Point` or `Bugis` |
| `type` | The LLM-assigned broad category, such as location, organization, or concept |
| `description` | A generated description based on how the entity appears in the input messages |
| `text_unit_ids` | References to the original SMS-derived text chunks supporting the entity |
| `degree` / `frequency` | How connected or recurrent the entity is in the graph, depending on GraphRAG version |

Think of an entity as: **“a named concept GraphRAG has learned from the SMS corpus.”**

### `relationships.parquet`

Each row is an **edge** connecting two entity nodes.

| Field | Meaning |
|---|---|
| `source` | The starting entity |
| `target` | The connected entity |
| `description` | The LLM-generated explanation of the connection |
| `weight` | The strength or frequency of the connection |
| `text_unit_ids` | Source text chunks supporting the relationship |

For example, if a message mentions Jurong Point and Bugis together, GraphRAG may create a relationship because those two locations co-occur in the same promotional SMS.

### `text_units.parquet`

These are the chunks of original input text used during graph extraction. They are the most direct bridge back to the SMS messages and are useful for checking whether entities and relationships are grounded in the source data.

### `community_reports.parquet`

GraphRAG groups connected entities into **communities** (topic clusters), then generates reports that summarize them. These reports are especially important for global search.

A community may capture themes such as:

- Singapore shopping and entertainment promotions
- Prize or marketing messages
- Financial scams and account-verification tactics, if sufficient examples exist in the input

## Interpreting results carefully

GraphRAG's entities, relationship descriptions, and community reports are generated interpretations of the input text. They are useful for discovery and retrieval, but they are not independently verified facts.

For the fraud/smishing project:

- Treat `text_units.parquet` as the evidence layer.
- Treat entity and relationship rows as hypotheses or structured summaries extracted from that evidence.
- Check high-impact claims against the original SMS text.
- Track cases where the model over-generalizes, misses a relevant connection, or gives an unsupported answer.

This is also useful evaluation data for a future secure KG-RAG gateway.

## Troubleshooting

### `Connection error` or `Cannot connect to host localhost:8686`

The embeddings server is not running, has crashed, or uses a different port/path.

Fix:

```bash
python -m uvicorn local_embeddings_server:app --host 127.0.0.1 --port 8686
```

Then confirm:

```bash
curl -v http://127.0.0.1:8686/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["test"], "model": "text-embedding-3-small"}'
```

### HTTP 404 from `GET /`

This is normal if the FastAPI application defines no root endpoint. The relevant endpoint is:

```text
POST /v1/embeddings
```

### HTTP 422: `Input should be a valid list`

Send a list of strings:

```json
{"input": ["test"], "model": "text-embedding-3-small"}
```

not:

```json
{"input": "test", "model": "text-embedding-3-small"}
```

### Metal / MPS crash on macOS

An error resembling the following indicates a macOS Metal/MPS crash:

```text
failed assertion _status < MTLCommandBufferStatusCommitted
zsh: abort python -m uvicorn ...
```

Use CPU explicitly when creating the SentenceTransformers model:

```python
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
```

This is slower than Apple GPU acceleration but is more stable for this local prototype.

### Groq model not found

If Groq returns a model-not-found error, choose a Groq model that your account can access, for example:

```yaml
model: groq/llama-3.1-8b-instant
```

Do not assume every model listed in provider documentation is enabled for every account.

### Query answers unrelated to the intended domain

If GraphRAG discusses unexpected content, such as shopping malls or cinemas, that may actually be content in the indexed SMS sample. Confirm this by checking `text_units.parquet` and `entities.parquet`.

Also verify that the input directory contains only the intended dataset. Re-index after removing old demo files or unrelated CSV/TXT files.

## Current status

The end-to-end pipeline has been successfully validated and run:

- Groq is used for graph extraction and community reporting.
- `all-MiniLM-L6-v2` is served locally for embeddings.
- The local endpoint returns `200 OK` for `/v1/embeddings`.
- GraphRAG indexing completed successfully.
- Local queries retrieved grounded content from the SMS spam CSV, including an example containing Jurong Point, Bugis, Great World, Cine, and Amore Wat.

## Next project steps

1. Expand beyond the 38-row test subset with a larger SMS spam/smishing corpus.
2. Ensure the corpus includes financial-fraud and smishing examples relevant to the threat model.
3. Build a small evaluation set of local and global questions.
4. Compare a baseline vector RAG system against GraphRAG on the same corpus.
5. Add a FastAPI-based secure KG-RAG gateway with logging, policy checks, grounding/consistency checks, and safe abstention.
6. Evaluate prompt-injection, document/KG-poisoning, hallucination, and unsafe financial-advice failure modes.

## Security notes

- Keep `GROQ_API_KEY` in `.env`; add `.env` to `.gitignore`.
- Do not upload real personal, financial, or sensitive SMS data to public viewers or public repositories.
- Clearly label any synthetic smishing examples used for evaluation.
- Do not treat model-extracted entities or relationships as verified facts without checking the original text.
