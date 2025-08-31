import logging
import os
import asyncio
import base64
from dotenv import load_dotenv

from langgraph.pregel.remote import RemoteGraph
from livekit.agents import Agent, AgentSession, get_job_context
from livekit.plugins import deepgram, silero
from livekit.plugins.turn_detector.english import EnglishModel
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    RoomInputOptions,
)
from livekit.agents.llm import ChatContext, ImageContent, ChatMessage
from livekit import rtc

from .adapter.langgraph import LangGraphAdapter

# Turn detector models: EnglishModel (light) / MultilingualModel (heavier)
# Ref: https://github.com/livekit/agents/blob/main/livekit-plugins/livekit-plugins-turn-detector/README.md
# Deepgram STT / TTS: https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-deepgram

load_dotenv(dotenv_path=".env")
logger = logging.getLogger("voice-agent")


class VisionAssistant(Agent):
    """Enhanced agent with vision capabilities for processing camera and screen sharing."""
    
    def __init__(self) -> None:
        self._latest_frame = None
        self._video_stream = None
        self._tasks = []
        self._screen_share_active = False
        self._has_video_input = False
        self._room = None
        super().__init__(
            instructions="You are a helpful voice AI assistant that helps manage todo lists. You can also see and understand visual content from the user's camera or screen sharing."
        )

    async def on_enter(self):
        """Initialize video processing when the agent enters the room."""
        logger.info("VisionAssistant on_enter called - initializing video handlers")

        # Attach to room events per official docs to guarantee we catch subscriptions
        try:
            room = get_job_context().room
            self._room = room

            @room.on("track_subscribed")
            def _room_on_track_subscribed(track, publication, participant):
                try:
                    logger.info(
                        f"room.on(track_subscribed) track={getattr(track,'sid',None)} kind={getattr(getattr(track,'kind',None),'value',getattr(track,'kind',None))} "
                        f"source={getattr(publication,'source',None)} participant={getattr(participant,'identity',None)}"
                    )
                    if getattr(track, "kind", None) == rtc.TrackKind.KIND_VIDEO:
                        self._create_video_stream(track, getattr(publication, "source", None))
                except Exception as e:
                    logger.warning(f"room.on(track_subscribed) handler error: {e}")
        except Exception as e:
            logger.debug(f"Unable to bind room track_subscribed handler: {e}")

        async def _delayed_scan_for_tracks():
            try:
                await asyncio.sleep(0.5)
                await self._attach_existing_remote_video_tracks()
            except Exception as e:
                logger.debug(f"Deferred scan for existing video tracks failed: {e}")

        asyncio.create_task(_delayed_scan_for_tracks())

    def _setup_video_callbacks(self):
        """Set up callbacks for video processing that the LiveKit framework can call."""
        logger.info("Setting up video processing callbacks")
        
        # The LiveKit framework will call these methods when video events occur
        # We need to implement them in a way that the framework can find them
        
        # Set up a method to handle video input streams
        # This will be called when the LiveKit framework detects video input
        self._handle_video_input = self._process_video_input
        
        # Add a method that the LiveKit framework can actually call
        # This method name should match what the framework expects
        self._on_video_stream_start = self._handle_video_stream_start
        
    def _handle_video_stream_start(self, participant: str, source: str):
        """Handle when the LiveKit framework starts reading a video stream."""
        logger.info(f"Video stream started: participant={participant}, source={source}")
        if source == "SOURCE_SCREENSHARE":
            self._screen_share_active = True
            self._has_video_input = True
            logger.info("Screen sharing stream started and processed")
        elif source == "SOURCE_CAMERA":
            self._has_video_input = True
            logger.info("Camera stream started and processed")
        
        # Try to get the actual video track from the room
        # This is the key - we need to access the actual video data
        try:
            # The framework has started reading the stream, so we should be able to access it
            logger.info(f"Video stream {source} is now available for processing")
        except Exception as e:
            logger.error(f"Error accessing video stream: {e}")
        
    def _process_video_input(self, source: str):
        """Handle video input streams from the LiveKit framework."""
        logger.info(f"Processing video input: {source}")
        if source == "SOURCE_SCREENSHARE":
            self._screen_share_active = True
            self._has_video_input = True
            logger.info("Screen sharing input detected and processed")
        elif source == "SOURCE_CAMERA":
            self._has_video_input = True
            logger.info("Camera input detected and processed")
        
        # Also handle the video stream attachment events
        # This will be called when the LiveKit framework attaches video streams
        self._handle_video_stream_attachment = self._process_video_stream_attachment
        
    def _process_video_stream_attachment(self, participant: str, source: str):
        """Handle video stream attachment events from the LiveKit framework."""
        logger.info(f"Video stream attached: participant={participant}, source={source}")
        if source == "SOURCE_SCREENSHARE":
            self._screen_share_active = True
            self._has_video_input = True
            logger.info("Screen sharing stream attached and processed")
        elif source == "SOURCE_CAMERA":
            self._has_video_input = True
            logger.info("Camera stream attached and processed")
        
    def on_track_subscribed(self, *args, **kwargs):
        """Handle video tracks when they're subscribed (robust to signature differences)."""
        try:
            track = None
            publication = None
            participant = None

            # Try kwargs first
            track = kwargs.get("track", None)
            publication = kwargs.get("publication", None)
            participant = kwargs.get("participant", None)

            # Fallback: infer from positional args by attribute presence
            if track is None or publication is None:
                for obj in args:
                    if hasattr(obj, "kind") and hasattr(obj, "sid"):
                        track = obj
                    elif hasattr(obj, "source"):
                        publication = obj
                    elif hasattr(obj, "identity"):
                        participant = obj

            # Log what we inferred
            logger.info(
                f"on_track_subscribed invoked (track={getattr(track,'sid',None)}, "
                f"kind={getattr(getattr(track,'kind',None),'value',getattr(track,'kind',None))}, "
                f"source={getattr(publication,'source',None)}, "
                f"participant={getattr(participant,'identity',None)})"
            )

            if track and getattr(track, "kind", None) == rtc.TrackKind.KIND_VIDEO:
                self._create_video_stream(track, getattr(publication, "source", None))
        except Exception as e:
            logger.warning(f"on_track_subscribed handler error: {e}")

    def on_video_input_available(self, source: str):
        """Handle when video input becomes available."""
        logger.info(f"Video input available: {source}")
        if source == "SOURCE_SCREENSHARE":
            self._screen_share_active = True
            logger.info("Screen sharing input detected")
        elif source == "SOURCE_CAMERA":
            logger.info("Camera input detected")
        
        # Set a flag to indicate we have video input
        self._has_video_input = True

    async def _wait_and_process_track(self, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        """Wait for a track to be subscribed and then process it."""
        try:
            # Wait a bit for the track to be subscribed
            await asyncio.sleep(0.5)
            
            # Check if the track is now subscribed
            if publication.track and publication.track.subscribed:
                logger.info(f"Track {publication.track.sid} is now subscribed, processing...")
                self._create_video_stream(publication.track, publication.source)
            else:
                logger.info(f"Track {publication.track.sid} is not yet subscribed")
        except Exception as e:
            logger.error(f"Error waiting for track: {e}")

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """Add visual context to the conversation when available."""
        if self._latest_frame:
            try:
                # Pass the actual frame as ImageContent so LLM can see it (per docs)
                new_message.content.append(ImageContent(image=self._latest_frame))
                logger.info("Added latest video frame to conversation context")
            except Exception as e:
                logger.warning(f"Failed to process video frame: {e}")
            finally:
                self._latest_frame = None
        elif hasattr(self, '_has_video_input') and self._has_video_input:
            # Add information about available video input
            video_info = "Screen sharing" if self._screen_share_active else "Camera"
            logger.info(f"Video input available: {video_info}")
            new_message.content.append(f"I see {video_info.lower()} input is available.")

    def _create_video_stream(self, track: rtc.Track, source: rtc.TrackSource | int | str | None):
        """Create a video stream to process frames from the track."""
        if self._video_stream is not None:
            try:
                asyncio.create_task(self._video_stream.aclose())
            except Exception:
                pass

        self._video_stream = rtc.VideoStream(track)
        logger.info(f"Created VideoStream for track {getattr(track,'sid',None)}")
        
        # Update screen share status - accept enum, numeric, or string
        try:
            src_val = source
            if isinstance(source, rtc.TrackSource):
                src_val = source
            elif isinstance(source, str):
                src_val = source
            elif isinstance(source, int):
                src_val = source
        except Exception:
            src_val = source

        if src_val in (rtc.TrackSource.SCREEN_SHARE if hasattr(rtc.TrackSource, 'SCREEN_SHARE') else None, 1, "SOURCE_SCREENSHARE"):
            self._screen_share_active = True
            logger.info("Screen sharing track detected")
        elif src_val in (rtc.TrackSource.CAMERA if hasattr(rtc.TrackSource, 'CAMERA') else None, 0, "SOURCE_CAMERA"):
            logger.info("Camera track detected")
        else:
            logger.info(f"Video track from unknown source: {source}")

        async def read_stream():
            try:
                async for event in self._video_stream:
                    self._latest_frame = event.frame
            except Exception as e:
                logger.error(f"Error reading video stream: {e}")
            finally:
                if self._video_stream:
                    await self._video_stream.aclose()

        task = asyncio.create_task(read_stream())
        task.add_done_callback(lambda t: self._tasks.remove(t))
        self._tasks.append(task)

    async def _attach_existing_remote_video_tracks(self):
        """Scan the current room for any existing remote video tracks and attach streams."""
        try:
            room = self._room or getattr(getattr(self, "session", None), "_room", None) or getattr(getattr(self, "session", None), "room", None)
            if room is None:
                logger.debug("No room available on agent to scan for video tracks")
                return

            # Iterate remote participants and their publications
            for _, participant in getattr(room, "remote_participants", {}).items():
                pubs = getattr(participant, "track_publications", {})
                for _, pub in pubs.items():
                    track = getattr(pub, "track", None)
                    if track and getattr(track, "kind", None) == rtc.TrackKind.KIND_VIDEO:
                        logger.info(
                            f"Found existing remote video track {getattr(track,'sid',None)} from participant {getattr(participant,'identity',None)}"
                        )
                        self._create_video_stream(track, getattr(pub, "source", None))
                        return
            logger.debug("No existing remote video tracks found during scan")
        except Exception as e:
            logger.debug(f"Error while scanning for existing video tracks: {e}")

    async def _process_video_frame(self, frame: rtc.VideoFrame) -> ImageContent | None:
        """Process a video frame and convert it to a format suitable for LLM processing."""
        try:
            logger.info(f"Processing video frame: {frame.width}x{frame.height}, type: {frame.type}")
            
            # Convert frame to BGRA format for processing
            frame_bgra = frame.convert(rtc.VideoBufferType.BGRA)
            logger.info(f"Converted frame to BGRA: {frame_bgra.width}x{frame_bgra.height}")
            
            # Create a basic image content with frame metadata
            frame_info = f"Video frame: {frame.width}x{frame.height}, format: {frame.type}"
            if self._screen_share_active:
                frame_info += " (Screen sharing)"
            else:
                frame_info += " (Camera)"
            
            logger.info(f"Frame info: {frame_info}")
            
            # For now, we'll add the frame info as text content since we can't encode the image
            # The LLM will receive this information about what the user is showing
            return ImageContent(
                image=f"data:text/plain;base64,{base64.b64encode(frame_info.encode()).decode()}"
            )
            
        except Exception as e:
            logger.error(f"Error processing video frame: {e}")
            return None

    async def cleanup(self):
        """Clean up video processing resources."""
        if self._video_stream is not None:
            await self._video_stream.aclose()
        
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        self._tasks.clear()


def prewarm(proc: JobProcess):
    """Preload VAD model to reduce cold-start latency."""
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    """LiveKit worker entrypoint with video processing capabilities.

    - Connect to the room with video enabled for processing camera and screen sharing.
    - Wait for a participant and optionally extract a LangGraph thread_id from metadata.
    - Create a RemoteGraph client to a running LangGraph server (dev or remote).
    - Start AgentSession wired to VAD, STT/TTS, LLM adapter, turn detection, and video processing.

    References:
    - LiveKit AgentSession: https://github.com/livekit/agents/blob/main/README.md
    - LangGraph RemoteGraph: https://github.com/langchain-ai/langgraph/blob/main/docs/docs/how-tos/use-remote-graph.md
    """
    logger.info(f"connecting to room {ctx.room.name} with video processing enabled")
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"participant: {participant}")

    # Use metadata if present, otherwise let LangGraph handle state
    # thread_id = participant.metadata if participant.metadata else None
    thread_id = "2432c116-7868-4d4a-8a87-189c58962dca"
    if thread_id:
        logger.info(f"Using threadId from metadata: {thread_id}")
    else:
        logger.info("No threadId provided - LangGraph will create new conversation state")

    logger.info(
        f"starting vision-enabled voice assistant for participant {participant.identity} (thread ID: {thread_id or 'new'})"
    )

    # LangGraph dev server URL (override via LANGGRAPH_URL)
    langgraph_url = os.getenv("LANGGRAPH_URL", "http://localhost:2024")

    # Remote LangGraph compiled graph. Ensure the LangGraph server is running.
    # RemoteGraph: https://github.com/langchain-ai/langgraph/blob/main/docs/docs/how-tos/use-remote-graph.md
    graph = RemoteGraph("todo_agent", url=langgraph_url)

    # Create the agent session with video processing enabled
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

    # Start the agent session with VisionAssistant for video processing
    vision_agent = VisionAssistant()
    # Provide room reference for fallback scanning
    try:
        vision_agent._room = ctx.room
    except Exception:
        pass
    await session.start(
        agent=vision_agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            video_enabled=True,
            audio_enabled=True,
            text_enabled=True
        )
    )

    # Initial greeting with interruptions allowed for responsiveness.
    await session.say(
        "Hey, I'm your vision-enabled AI assistant! I can help you manage your todo list and I can see what you're showing me through your camera or screen sharing.",
        allow_interruptions=True,
    )

    # Keep the session running to process video streams
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    finally:
        # Clean up video processing resources
        if hasattr(session, 'agent') and hasattr(session.agent, 'cleanup'):
            await session.agent.cleanup()


if __name__ == "__main__":
    # Standard worker runner. Use `uv run -m src.livekit.agent console` for local terminal I/O.
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
