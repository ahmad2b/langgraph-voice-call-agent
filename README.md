# LangGraph Voice Call Agent (LiveKit)

A real-time voice/call AI agent that lets you talk to a LangGraph agent over LiveKit — similar to "voice mode" experiences in ChatGPT Voice, OpenAI Realtime API sessions, and Gemini Live. This repo demonstrates adapting any LangGraph agent into a full-duplex, low-latency voice assistant using LiveKit Agents.

## Project Structure

```
langgraph-voice-call-agent/
├── src/                          # Main source code
│   ├── livekit/                  # LiveKit agent implementation
│   │   ├── agent.py             # Main agent entrypoint
│   │   └── adapter/             # LangGraph integration
│   │       └── langgraph.py     # LangGraph adapter for LiveKit
│   └── langgraph/               # LangGraph definitions
│       └── agent.py             # Todo management agent
├── compose.yml                   # Docker Compose for local LiveKit server
├── pyproject.toml               # Python project configuration
├── uv.lock                      # uv dependency lock file
└── Makefile                     # Development commands
```

## Quick Start

### Prerequisites

- **Python 3.11+** with `uv` package manager
- **Docker & Docker Compose** for local LiveKit server
- **LiveKit Cloud account** (optional, for cloud deployment)

### Installation

1. **Clone and setup the project:**
```bash
git clone https://github.com/ahmad2b/langgraph-voice-call-agent.git
cd langgraph-voice-call-agent

# Initialize with uv
uv sync
```

2. **Download required model files:**
```bash
make download-files
# or
uv run -m src.livekit.agent download-files
```

3. **Start local LiveKit server:**
```bash
docker compose up -d
```

4. **Run the agent:**
```bash
make dev
# or
uv run -m src.livekit.agent dev
```

## 🔧 Development Setup

### Using `uv` (Recommended)

This project uses `uv` for fast Python package management:

```bash
# Install dependencies
uv sync

# Add new dependencies
uv add package-name

# Add dev dependencies
uv add --dev package-name

# Run commands
uv run -m src.livekit.agent dev
uv run -m src.livekit.agent download-files
```

## Local Development

### Local LiveKit Server

The `compose.yml` provides a local LiveKit server for development:

```yaml
# Key configuration:
- Port 7880: API and WebSocket
- Port 7881: TURN/TLS
- Port 7882: UDP for media
- Development keys: "devkey: secret"
```

**Start local server:**
```bash
docker compose up -d
```

**Check server status:**
```bash
docker compose ps
docker compose logs livekit
```

### LangGraph Dev Server (Required)

Run the LangGraph API server locally so the LiveKit agent can call your graph via RemoteGraph.

```bash
# Python CLI (default port 2024)
uv run langgraph dev
```

Set the LangGraph server URL (optional; defaults to http://localhost:2024):

```bash
# .env
LANGGRAPH_URL=http://localhost:2024
```

The agent reads `LANGGRAPH_URL` and falls back to `http://localhost:2024` if not set.

### Environment Variables

Create `.env` file for local development:

```bash
# LiveKit Local Server
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret

# OpenAI (for LangGraph agent)
OPENAI_API_KEY=your-openai-key

# Deepgram (for STT/TTS)
DEEPGRAM_API_KEY=your-deepgram-key

# LangGraph dev server (optional; default http://localhost:2024)
LANGGRAPH_URL=http://localhost:2024

# Thread Management (optional)
# Pass threadId via participant metadata for conversation continuity
# If no metadata provided, LangGraph will create new conversation state
```

### Notes on Audio Filters / Noise Cancellation

- Enhanced noise cancellation and Cloud audio filters are a LiveKit Cloud feature.
- On local servers, you may see warnings. This is expected.

## ☁️ LiveKit Cloud Deployment

For production use, deploy to LiveKit Cloud for better performance and features.

### 1. Get LiveKit Cloud Credentials

1. Sign up at [LiveKit Cloud](https://cloud.livekit.io/)
2. Create a new project
3. Get your API keys from the project dashboard

### 2. Update Environment Variables

```bash
# LiveKit Cloud
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your-api-key
LIVEKIT_API_SECRET=your-api-secret
```

### 3. Update Agent Configuration

Modify `src/livekit/agent.py` to use cloud URL:

```python
# For cloud deployment, remove local server setup
# The agent will connect to LiveKit Cloud automatically
```

## 📁 File Descriptions

### Core Files

- **`src/livekit/agent.py`**: Main LiveKit agent entrypoint
  - Connects to LiveKit room
  - Manages participant sessions
  - Integrates VAD, STT, LLM, TTS, and turn detection
  - Extracts threadId from participant metadata for conversation continuity

- **`src/livekit/adapter/langgraph.py`**: LangGraph integration adapter
  - Bridges LiveKit LLM interface to LangGraph workflows
  - Handles streaming responses (`messages` and `custom` modes)
  - Converts LangGraph outputs to LiveKit ChatChunks

- **`src/langgraph/agent.py`**: Todo management agent
  - Defines ReAct agent with todo tools
  - Handles add, list, complete, and delete operations
  - Supports user confirmation for deletions

### Configuration Files

- **`compose.yml`**: Local LiveKit server setup
- **`pyproject.toml`**: Python project configuration
- **`Makefile`**: Development commands and shortcuts

## 🔌 Plugin Integration

### Speech Processing

- **VAD (Voice Activity Detection)**: Silero VAD for speech detection
- **STT (Speech-to-Text)**: Deepgram for transcription
- **TTS (Text-to-Speech)**: Deepgram for speech synthesis
- **Turn Detection**: English model for conversation flow

### AI Integration

- **LangGraph Agent**: Todo management with ReAct agent
- **LLM**: OpenAI GPT-4 for reasoning and responses
- **Streaming**: Real-time response generation

## 🚦 Available Commands

```bash
# Development
make dev                    # Run agent in development mode
make langgraph-dev          # Run the LangGraph dev server (uv run langgraph dev)
make dev-all                # Start LangGraph server and LiveKit agent together
make download-files         # Download required model files
make clean                  # Clean up cache files
make help                   # Show available commands

# Direct execution
uv run -m src.livekit.agent dev
uv run -m src.livekit.agent download-files
```

## 🌐 Testing the Agent

### Frontend Options

1. **LiveKit Examples**: Use frontends from [livekit-examples](https://github.com/livekit-examples/)
2. **Custom Frontend**: Build your own using [LiveKit client SDKs](https://docs.livekit.io/reference/client-sdks/)
3. **LiveKit Sandbox**: Test instantly with [LiveKit Cloud Sandbox](https://cloud.livekit.io/projects/p_/sandbox)

### Connection Details

- **Local**: `ws://localhost:7880`
- **Cloud**: `wss://your-project.livekit.cloud`
- **Room**: Auto-generated room names
- **Authentication**: API key/secret or JWT tokens

## 🔧 Troubleshooting

### Common Issues

1. **Model Download Issues**: VAD and turn detection models need downloading
   - Solution: Run `make download-files` first

2. **Port Conflicts**: LiveKit ports already in use
   - Solution: Check `docker compose ps` and stop conflicting services

3. **Import Errors**: Module not found
   - Solution: Use `uv run -m src.livekit.agent` instead of direct file execution

## 📚 References

- [LiveKit Agents Documentation](https://github.com/livekit/agents)
- [LiveKit Self-Hosting Guide](https://docs.livekit.io/home/self-hosting/)
- [LiveKit Cloud Documentation](https://docs.livekit.io/home/cloud/)
- [LangGraph Documentation](https://github.com/langchain-ai/langgraph)
- [LiveKit Client SDKs](https://docs.livekit.io/reference/client-sdks/)

## 🤝 Contributing

This project demonstrates LiveKit + LangGraph integration patterns. Feel free to:

- Report issues
- Suggest improvements
- Submit pull requests
- Use as a reference for your own projects

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

Built with ❤️ using [LiveKit](https://livekit.io/) and [LangGraph](https://github.com/langchain-ai/langgraph)

## 🙏 Acknowledgments

Based on and inspired by [dqbd/langgraph-livekit-agents](https://github.com/dqbd/langgraph-livekit-agents).