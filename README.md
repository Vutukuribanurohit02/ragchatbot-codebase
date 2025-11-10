# RAG Chatbot Codebase

A production-ready Retrieval-Augmented Generation (RAG) system designed to answer questions about course materials using semantic search and AI-powered responses.

## Overview

This application is a full-stack web application that enables users to query course materials and receive intelligent, context-aware responses. It leverages ChromaDB for vector storage, Anthropic's Claude for AI generation, and provides an intuitive web interface for seamless interaction.

## Features

- **Semantic Search**: Advanced vector-based search using ChromaDB for accurate content retrieval
- **AI-Powered Responses**: Integration with Anthropic's Claude (Sonnet 4) for intelligent answer generation
- **Tool-Based Architecture**: Modular search tools that the AI can leverage dynamically
- **Course Management**: Automatic processing and indexing of course materials (PDF, DOCX, TXT)
- **Conversation History**: Session-based conversation management for contextual responses
- **Real-Time Statistics**: Live course and content statistics displayed in the UI
- **Source Attribution**: Transparent sourcing with references to original course materials
- **Responsive UI**: Clean, modern interface built with vanilla JavaScript

## Architecture

### Backend (FastAPI)
- **RAGSystem**: Main orchestrator coordinating all components
- **VectorStore**: ChromaDB-based dual-collection storage (catalog + content)
- **AIGenerator**: Claude API integration with tool-calling capabilities
- **Search Tools**: Intelligent search system with course name resolution
- **DocumentProcessor**: Automatic document chunking and metadata extraction

### Frontend (Vanilla JavaScript)
- Single-page application with markdown rendering
- Real-time course statistics and suggested queries
- Responsive design with collapsible sidebar

### Data Flow
1. Documents loaded from `docs/` directory on startup
2. Content chunked and stored in ChromaDB collections
3. User queries processed through AI with search tool access
4. Results displayed with source attribution

## Prerequisites

- Python 3.13 or higher
- uv (Python package manager)
- An Anthropic API key

## Quick Start

### 1. Install Dependencies

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python dependencies
uv sync
```

### 2. Configure Environment

Create a `.env` file in the `ragchatbot-codebase-main` directory:

```bash
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### 3. Run the Application

```bash
cd ragchatbot-codebase-main
chmod +x run.sh
./run.sh
```

Or manually:

```bash
cd ragchatbot-codebase-main/backend
uv run uvicorn app:app --reload --port 8000
```

### 4. Access the Application

- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## Project Structure

```
ragchatbot-codebase-main/
├── backend/              # FastAPI backend application
│   ├── app.py           # Main FastAPI application
│   ├── rag_system.py    # RAG orchestration
│   ├── vector_store.py  # ChromaDB integration
│   ├── ai_generator.py  # Claude API integration
│   ├── search_tools.py  # Search tool implementations
│   ├── config.py        # Configuration settings
│   └── chroma_db/       # Persistent vector storage
├── frontend/            # Vanilla JavaScript frontend
│   ├── index.html       # Main HTML file
│   ├── styles.css       # Styling
│   └── script.js        # Application logic
├── docs/                # Course materials directory
├── scripts/             # Utility scripts
│   ├── format.sh        # Code formatting (modifies files)
│   └── lint.sh          # Code quality checks (read-only)
├── pyproject.toml       # Python dependencies
└── run.sh              # Quick start script
```

## Development

### Code Quality Tools

Install development dependencies:

```bash
uv sync --group dev
```

**Format code** (automatically fixes issues):

```bash
./scripts/format.sh
```

**Lint code** (checks without modifications):

```bash
./scripts/lint.sh
```

### Adding Course Materials

1. Place course documents (PDF, DOCX, or TXT) in the `docs/` directory
2. Restart the application
3. Documents are automatically processed and indexed

## Configuration

Key settings in `backend/config.py`:

- **Chunk size**: 800 characters with 100 character overlap
- **Embedding model**: all-MiniLM-L6-v2 (SentenceTransformers)
- **Max search results**: 5 per query
- **Conversation history**: 2 message pairs
- **AI Model**: claude-sonnet-4-20250514

## API Endpoints

- `GET /` - Serve frontend interface
- `POST /api/query` - Submit a query and get AI-generated response
- `GET /api/stats` - Get course and content statistics
- `GET /api/suggested-queries` - Get suggested example queries
- `GET /docs` - Interactive API documentation (Swagger UI)

## Technologies

- **Backend**: FastAPI, ChromaDB, SentenceTransformers, Anthropic Claude API
- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **Package Management**: uv
- **Vector Database**: ChromaDB with persistent storage
- **AI Model**: Claude Sonnet 4

## Troubleshooting

### Scripts not executable
```bash
chmod +x scripts/*.sh run.sh
```

### ChromaDB persistence issues
Delete and reinitialize:
```bash
rm -rf backend/chroma_db/
# Restart application to rebuild
```

### Environment variables not loading
Ensure `.env` file is in `ragchatbot-codebase-main/` directory, not root.

## Contributing

See `CLAUDE.md` for detailed development guidelines and architecture documentation.

## License

This project is provided as-is for educational and development purposes.
