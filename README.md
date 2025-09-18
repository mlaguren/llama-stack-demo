# llama-stack-demo

A demo project showcasing a scalable **LLM-powered stack** using:

- **FastAPI** → REST endpoints for ingestion & querying  
- **Ray** → orchestration and distributed task execution  
- **LlamaIndex** → indexing & retrieval for unstructured data  
- **Postgres** → vector database backend  
- **Groq LLM** → high-performance inference for embeddings & query responses  
- **Docker & Kubernetes** → containerized and deployable anywhere  

---

## Architecture
```mermaid
flowchart LR
  %% Endpoints outside
  QueryEp["/query endpoint"]
  IngestEp["/ingest endpoint"]

  %% FastAPI gateway
  subgraph APIServer[Uvicorn: API Server]
    F["FastAPI"]
  end
  QueryEp --> F
  IngestEp --> F

  %% LlamaIndex on Ray
  subgraph RayCluster[Ray Cluster]
    Serve["Ray Serve router"]
    R1["LlamaIndex replica A"]
    R2["LlamaIndex replica B"]
    Serve --> R1
    Serve --> R2
  end

  %% Vector DB
  subgraph llamastore[Embeddings Store]
    V["Postgres PGVector"]
  end

  %% Flows
  F <--> |/query and response| Serve
  F <--> |/ingest and ack| Serve

  R1 --> |store and retrieve embeddings| V
  R2 --> |store and retrieve embeddings| V


## Quick Start
```bash
git clone https://github.com/mlaguren/llama-stack-demo.git
cd llama-stack-demo
```

# set your API keys

Create .env file and add your GROQ API Key

```bash
GROQ_API_KEY=your_api_key_here
OPENAI_API_KEY=your_api_key
LANGSMITH_API_KEY=you_api_key
LANGSMITH_PROJECT=project_name
LANGSMITH_TRACING=true
```
Bring up llama-stack-demo
```bash
docker compose up --build
```
Services:

| Service           | URL / Port                                     | Notes                                                     |
| ----------------- | ---------------------------------------------- | --------------------------------------------------------- |
| **API**           | [http://localhost:8000](http://localhost:8000) | FastAPI ingestion & query endpoints                       |
| **Ray Dashboard** | [http://localhost:8265](http://localhost:8265) | Cluster status, logs, and **profiling (CPU flamegraphs)** |
| **Postgres**      | localhost:5432                                 | Vector DB with PGVector extension                         |


# Validate Environment

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "Llamas are awesome and work well in clusters."}'
```

Should get the following response:

```json
{
    "status": "success",
    "message": "Document ingested."
}
```
To validate your AI Keys:
```bash
curl -X POST http://localhost:8000/query \ 
  -H "Content-Type: application/json" \
  -d '{"question": "What animals work well in clusters?"}'
```

Should yield the following response:

```json
{"answer":"Llamas"}
```

Uploading a pdf

```bash
curl -X POST "http://localhost:8000/ingest_pdf" \
  -F "file=@path-to-pdf-file"
```

Response:
```json
{"status":"success","filename":"legends.pdf","pages_ingested":134,"chunks_ingested":138,"chars_ingested":269436}
```

Query PDF file
```bash
curl -X POST "http://localhost:8000/query" \          
     -H "Content-Type: application/json" \
     -d '{"question":"Summarize the Fox and the Wolf?"}'
```

Response
```json
{
    "answer": "The fox and the wolf lived together in the same den, with the wolf oppressing the fox. The fox advised the wolf to be kind and abandon wickedness, warning him of the cunning nature of humans. The wolf rejected the advice and struck the fox, who apologized and recited verses seeking forgiveness. The wolf accepted the apology but warned the fox to not speak out of turn. The fox, realizing the wolf's intentions, decided to act with caution and dissimulation. Despite enduring mistreatment, the fox remained patient and cautious, ultimately outsmarting the wolf by avoiding a potential trap in a vineyard.",
    "sources": [
        {
            "node_id": "b6829e9a-38f1-4bb9-88ee-59607745308e",
            "score": 0.6338359088155088,
            "filename": "legends.pdf",
            "page_number": 106
        },
        {
            "node_id": "210f70c3-91c4-423d-8237-0d14f445df2c",
            "score": 0.5692067999120484,
            "filename": "legends.pdf",
            "page_number": 107
        }
    ]
}
```

# Infra: Custom Ray Image with Profiling

The Ray Dashboard profiling tools (CPU flamegraphs, usage reports) require py-spy with root permissions.
This project includes a custom image in infra/ray/Dockerfile
 that:

* Builds and installs py-spy (compiled via Rust for ARM64).
* Configures it with setuid root so it can attach to Ray worker processes.
* Provides a minimal sudo shim so Ray can invoke it.
* Symlinks it to /home/ray/anaconda3/bin/py-spy where the dashboard expects it.

docker-compose.yml is already configured to use this image and sets the necessary capabilities:
```yaml
cap_add:
  - SYS_PTRACE
security_opt:
  - seccomp=unconfined
  - apparmor=unconfined
```

## Troubleshooting

### Flamegraph shows “Permission denied (os error 13)”
* Ensure you built the custom Ray image: docker compose build ray
* Confirm inside the container:
```bash
docker exec -it llama-ray bash -lc 'which py-spy && ls -l $(which py-spy) && sudo -n py-spy --version'
```
### services.container_name must be a mapping
* Check your docker-compose.yml for duplicate or mis-indented ray: blocks. Only one ray service should exist under services:.

### apt-get exit 100 during build
* The Ray base image on ARM doesn’t always support apt installs cleanly. The provided multi-stage infra/ray/Dockerfile avoids this by compiling py-spy with Cargo.