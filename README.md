# LangGraph Voice Call Agent

A real-time voice/call AI agent that lets you talk to a LangGraph agent over LiveKit, similar to "voice mode" experiences in ChatGPT Voice, OpenAI Realtime API sessions, and Gemini Live. This repo demonstrates adapting any LangGraph agent into a full-duplex, low-latency voice assistant using LiveKit's real-time communication infrastructure.

This backend provides the core voice processing and AI agent functionality, built to work seamlessly with LiveKit's real-time infra and any frontend that supports LiveKit client connections.

## Features

- **Real-time voice interaction** with LangGraph agents
- **Full-duplex communication** with low-latency audio processing
- **Flexible LangGraph integration** - works with any LangGraph agent
- **Comprehensive audio pipeline** including VAD, STT, TTS, and turn detection
- **Thread-based conversation continuity** via participant metadata

## Project Structure

```
langgraph-voice-call-agent/
├── src/                         # Main source code
│   ├── livekit/                 # LiveKit agent implementation
│   │   ├── agent.py             # Main agent entrypoint
│   │   └── adapter/             # LangGraph integration
│   │       └── langgraph.py     # LangGraph adapter for LiveKit
│   └── langgraph/               # LangGraph Agent Sdefinitions
│       └── agent.py             # An example agent
├── compose.yml                  # Docker Compose for local LiveKit server
├── pyproject.toml               # Python project configuration
├── uv.lock                      # uv dependency lock file
└── Makefile                     # Development commands
```

## How it works (high level)

1. **Agent Initialization** → LiveKit agent connects to room and waits for participants
2. **Audio Pipeline Setup** → VAD, STT, TTS, and turn detection models are loaded and configured  
3. **LangGraph Integration** → Connect to LangGraph server
4. **Voice Processing** → Real-time audio is processed through the pipeline:
   - Voice Activity Detection (VAD) detects when user speaks
   - Speech-to-Text (STT) transcribes audio to text
   - LangGraph agent processes the query and generates responses
   - Text-to-Speech (TTS) converts responses back to audio
   - Turn detection manages conversation flow
5. **Thread Continuity** → Conversation state is maintained via thread IDs from participant metadata

## Architecture

- **Backend**: Python with LiveKit Agents and LangGraph
- **Voice Infrastructure**: LiveKit's real-time infra
- **AI Agents**: LangGraph agents
- **Audio Pipeline**: Deepgram STT/TTS, Silero VAD, English turn detection
- **State Management**: Thread-based conversation continuity

## Quick Start

### Prerequisites

- **Python 3.12+** with `uv` package manager
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

## Development Setup

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
```

## LiveKit Cloud Deployment

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

## File Descriptions

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

## Testing the Agent

### Frontend

[LangGraph Voice Call Agent Web](https://github.com/ahmad2b/langgraph-voice-call-agent-web)

#### Using the [LangGraph Voice Call Agent Web](https://github.com/ahmad2b/langgraph-voice-call-agent-web)

1. Start this backend (see Quick Start above)
2. Clone and run the frontend:
   ```bash
   git clone https://github.com/ahmad2b/langgraph-voice-call-agent-web.git
   cd langgraph-voice-call-agent-web
   npm install && npm run dev
   ```
3. Open http://localhost:3000

### Connection Details

- **Local**: `ws://localhost:7880`
- **Cloud**: `wss://your-project.livekit.cloud`
- **Room**: Auto-generated room names
- **Authentication**: API key/secret or JWT tokens

## Troubleshooting

### Common Issues

#### 1. **Model Download Issues** 
VAD and turn detection models need downloading before first use.

**Error symptoms:**
```
FileNotFoundError: Model files not found
```

**Solution:**
```bash
make download-files
# or directly
uv run -m src.livekit.agent download-files
```

#### 2. **Port Conflicts**
LiveKit ports already in use.

**Solution:**
```bash
docker compose ps
docker compose down  # Stop existing containers
docker compose up -d
```

#### 3. **Import Errors**
Module not found errors.

**Solution:**
Always use the module format:
```bash
# ✅ Correct
uv run -m src.livekit.agent dev

# ❌ Incorrect  
python src/livekit/agent.py
```

#### 4. **LangGraph Connection Issues**
Agent can't connect to LangGraph server.

**Error symptoms:**
```
Connection refused to localhost:2024
```

**Solution:**
```bash
# Ensure LangGraph server is running
uv run langgraph dev

# Or run both together
make dev-all
```

#### 5. **Environment Variable Issues**
Missing or incorrect API keys.

**Solution:**
Create `.env` file with all required variables:
```bash
cp .env.example .env  # If available
# Then edit .env with your actual keys
```

### Getting Help

If you continue experiencing issues:

1. **Check logs** for specific error messages
2. **Verify system requirements** (Python 3.12+)
3. **Test with minimal setup** (local LiveKit server first)
4. **Check LiveKit Cloud status** if using cloud deployment

## References

- [LiveKit Agents Documentation](https://github.com/livekit/agents)
- [LiveKit Self-Hosting Guide](https://docs.livekit.io/home/self-hosting/)
- [LiveKit Cloud Documentation](https://docs.livekit.io/home/cloud/)
- [LangGraph Documentation](https://github.com/langchain-ai/langgraph)

## Contributing

This project is open source and welcome contributions! Please open a PR or issue through GitHub.

This project demonstrates LiveKit + LangGraph integration patterns. Feel free to:

- Report issues and bugs
- Suggest improvements and new features
- Submit pull requests
- Use as a reference for your own voice agent projects
- Share your own LangGraph agent implementations

## Connect

I'm actively exploring voice-first and real-time agents. If you're building in this space or experimenting with real-time AI infrastructure, I'd love to trade ideas, collaborate, or help out.

- GitHub: [ahmad2b](https://github.com/ahmad2b)  
- Twitter/X: [@mahmad2b](https://x.com/mahmad2b)  
- LinkedIn: [Ahmad Shaukat](https://www.linkedin.com/in/ahmad2b)  
- Book a chat: [cal.com/mahmad2b/15min](https://cal.com/mahmad2b/15min)

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

Inspired by [dqbd/langgraph-livekit-agents](https://github.com/dqbd/langgraph-livekit-agents).