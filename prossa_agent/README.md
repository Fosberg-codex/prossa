# Prossa Agent - RAG Framework

A modular Retrieval-Augmented Generation (RAG) framework built with multiple LLM support and efficient vector storage.

## Features

- **Multiple LLM Support**
  - OpenAI GPT-4o and GPT-o1-preview
  - Anthropic Claude 3.5 (Sonnet and Haiku)
  - Google Gemini Pro and Base
  
- **RAG Capabilities**
  - Efficient vector storage and retrieval
  - Context-aware response generation
  - Semantic similarity search
  - GPU-accelerated embeddings

- **Advanced Validation**
  - Context relevance checking
  - Response consistency validation
  - Confidence scoring
  - Cross-validation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/prossa-agent.git
cd prossa-agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create `.env` file from template:
```bash
cp .env.template .env
```

4. Add your API keys to `.env`:
```env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_GEMINI_API_KEY=your_gemini_key
```

## Usage

Basic usage example:

```python
from prossa_agent.core.prossa_agent import ProsaAgent, AgentConfig

# Initialize agent
config = AgentConfig(
    max_memory_items=1000,
    enable_gpu=True,
    min_confidence=0.7
)
agent = ProsaAgent(config)

# Process a query
async def main():
    response = await agent.process_query(
        "What are the key features of transformer architectures?"
    )
    print(response)

# Run
import asyncio
asyncio.run(main())
```

## Configuration

Key configuration options in `AgentConfig`:

- `max_memory_items`: Maximum number of items in memory buffer (default: 1000)
- `enable_gpu`: Whether to use GPU acceleration (default: True)
- `min_confidence`: Minimum confidence threshold (default: 0.7)
- `log_level`: Logging level (default: "INFO")

## Architecture

The framework follows a modular architecture:

```
prossa_agent/
├── core/               # Core components
├── llm/                # LLM implementations
├── memory/             # Vector storage
├── analysis/           # Task analysis
├── integrations/       # External integrations
└── utils/              # Utilities
```

## Key Components

1. **ProsaAgent**: Main agent class implementing the RAG workflow
2. **MemoryBuffer**: Vector storage and retrieval system
3. **ValidationEngine**: Response validation and confidence scoring
4. **ActionManager**: Manages text processing and response generation
5. **ModelSelector**: Intelligent LLM model selection

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [Prossa](https://pypi.org/project/prossa/) library
- Uses [sentence-transformers](https://www.sbert.net/) for embeddings
- Supports multiple LLM providers (OpenAI, Anthropic, Google)