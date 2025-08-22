import logging
import os
from dotenv import load_dotenv

from langgraph.pregel.remote import RemoteGraph
from livekit.agents import Agent, AgentSession
from livekit.plugins import deepgram, silero
from livekit.plugins.turn_detector.english import EnglishModel
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
)

from .adapter.langgraph import LangGraphAdapter

# Turn detector models: EnglishModel (light) / MultilingualModel (heavier)
# Ref: https://github.com/livekit/agents/blob/main/livekit-plugins/livekit-plugins-turn-detector/README.md
# Deepgram STT / TTS: https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-deepgram

load_dotenv(dotenv_path=".env")
logger = logging.getLogger("voice-agent")


def prewarm(proc: JobProcess):
    """Preload VAD model to reduce cold-start latency."""
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    """LiveKit worker entrypoint.

    - Connect to the room (audio-only autosubscribe for efficiency).
    - Wait for a participant and optionally extract a LangGraph thread_id from metadata.
    - Create a RemoteGraph client to a running LangGraph server (dev or remote).
    - Start AgentSession wired to VAD, STT/TTS, LLM adapter, and turn detection.

    References:
    - LiveKit AgentSession: https://github.com/livekit/agents/blob/main/README.md
    - LangGraph RemoteGraph: https://github.com/langchain-ai/langgraph/blob/main/docs/docs/how-tos/use-remote-graph.md
    """
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"participant: {participant}")

    # Use metadata if present, otherwise let LangGraph handle state
    thread_id = participant.metadata if participant.metadata else None
    if thread_id:
        logger.info(f"Using threadId from metadata: {thread_id}")
    else:
        logger.info("No threadId provided - LangGraph will create new conversation state")

    logger.info(
        f"starting voice assistant for participant {participant.identity} (thread ID: {thread_id or 'new'})"
    )

    # LangGraph dev server URL (override via LANGGRAPH_URL)
    langgraph_url = os.getenv("LANGGRAPH_URL", "http://localhost:2024")

    # Remote LangGraph compiled graph. Ensure the LangGraph server is running.
    # RemoteGraph: https://github.com/langchain-ai/langgraph/blob/main/docs/docs/how-tos/use-remote-graph.md
    graph = RemoteGraph("todo_agent", url=langgraph_url)

    # Create the agent session
    # AgentSession wiring & options: https://github.com/livekit/agents/blob/main/README.md#_snippet_1
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=LangGraphAdapter(graph, config={"configurable": {"thread_id": thread_id}} if thread_id else {}),
        tts=deepgram.TTS(),
        turn_detection=EnglishModel(),
        min_endpointing_delay=0.5,
        max_endpointing_delay=5.0,
    )

    # Start the agent session with simple instructions.
    await session.start(
        agent=Agent(
        instructions="You are a helpful voice AI assistant that helps manage todo lists."),
        room=ctx.room,
    )

    # Initial greeting with interruptions allowed for responsiveness.
    await session.say(
        "Hey, how can I help you today? I can help you manage your todo list.",
        allow_interruptions=True,
    )


if __name__ == "__main__":
    # Standard worker runner. Use `uv run -m src.livekit.agent console` for local terminal I/O.
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
