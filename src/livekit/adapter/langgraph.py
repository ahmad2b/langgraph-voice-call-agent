"""
LangGraphAdapter bridges LiveKit Agents' LLM interface to a LangGraph workflow.

Key ideas:
- We stream LangGraph outputs using stream_mode=["messages", "custom"].
- "messages" chunks are converted to LiveKit llm.ChatChunk with ChoiceDelta(content=str).
- "custom" chunks support simple events like {"type": "say", data: {content: str}}.

References:
- LiveKit Agents LLM API (ChatChunk, ChoiceDelta): docs/livekit/agents (repo README and llm module)
- LangGraph streaming modes: messages/custom and astream():
  https://github.com/langchain-ai/langgraph/blob/main/docs/docs/how-tos/streaming.md
- RemoteGraph astream usage:
  https://github.com/langchain-ai/langgraph/blob/main/docs/docs/how-tos/use-remote-graph.md
"""

from typing import Any, Optional
from httpx import HTTPStatusError
from livekit.agents import llm
from livekit.agents.types import (
    APIConnectOptions,
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import shortuuid
from livekit.agents.llm.tool_context import FunctionTool, RawFunctionTool, ToolChoice
from langgraph.pregel import Pregel
from langchain_core.messages import BaseMessageChunk, AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command
from langgraph.errors import GraphInterrupt

import logging

logger = logging.getLogger(__name__)


class LangGraphStream(llm.LLMStream):
    """LLMStream implementation that proxies a LangGraph stream.

    - Creates LiveKit ChatChunks from LangGraph "messages" stream chunks.
    - Passes through simple custom events (e.g., "say") from LangGraph "custom" stream.

    See:
      - LangGraph stream modes: https://github.com/langchain-ai/langgraph/blob/main/docs/docs/how-tos/streaming.md
      - LiveKit LLM stream contract: livekit.agents.llm.LLMStream (in repo)
    """

    def __init__(
        self,
        llm: llm.LLM,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[FunctionTool | RawFunctionTool],
        conn_options: APIConnectOptions,
        graph: Pregel,
    ):
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._graph = graph

    async def _run(self):
        """Consume LangGraph stream and emit LiveKit ChatChunks."""
        state = self._chat_ctx_to_state()

        # see if we need to respond to an interrupt (resume)
        if interrupt := await self._get_interrupt():
            used_messages = [AIMessage(interrupt.value)]
            # resume with last user content if any
            last_user = next(
                (m for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)),
                None,
            )
            if last_user:
                used_messages.append(last_user)
                input_state = Command(resume=(last_user.content, used_messages))
            else:
                input_state = Command(resume=(interrupt.value, used_messages))
        else:
            input_state = state

        try:
            # LangGraph astream with explicit modes (messages, custom)
            # https://github.com/langchain-ai/langgraph/blob/main/docs/docs/how-tos/streaming.md
            async for mode, data in self._graph.astream(
                input_state, config=self._llm._config, stream_mode=["messages", "custom"]
            ):
                if mode == "messages":
                    if data and len(data) > 0:
                        message = data[0]
                        if chunk := await self._to_livekit_chunk(message):
                            self._event_ch.send_nowait(chunk)

                if mode == "custom":
                    # Minimal custom protocol: {"type": "say", data: {content: str}}
                    if isinstance(data, dict) and (event := data.get("type")):
                        if event in ("say"):
                            content = (data.get("data") or {}).get("content")
                            if chunk := await self._to_livekit_chunk(content):
                                self._event_ch.send_nowait(chunk)
        except GraphInterrupt:
            # Graph was interrupted; we gracefully stop streaming
            pass

        # If interrupted late, send the string as a message
        if interrupt := await self._get_interrupt():
            if chunk := await self._to_livekit_chunk(interrupt.value):
                self._event_ch.send_nowait(chunk)

    def _chat_ctx_to_state(self) -> dict[str, Any]:
        """Translate LiveKit ChatContext into LangGraph state messages.

        We map LiveKit roles to LangChain message classes (AIMessage/HumanMessage/SystemMessage).
        """
        messages: list[AIMessage | HumanMessage | SystemMessage] = []
        for item in getattr(self._chat_ctx, "items", []):
            if getattr(item, "type", None) != "message":
                continue
            content = getattr(item, "text_content", None)
            if not content:
                continue
            role = getattr(item, "role", None)
            item_id = getattr(item, "id", None)
            if role == "assistant":
                messages.append(AIMessage(content=content, id=item_id))
            elif role == "user":
                messages.append(HumanMessage(content=content, id=item_id))
            elif role in ["system", "developer"]:
                messages.append(SystemMessage(content=content, id=item_id))

        return {"messages": messages}

    async def _get_interrupt(cls) -> Optional[str]:
        """Inspect graph state for latest assistant interrupt string.

        Uses Pregel.aget_state to retrieve interrupts from tasks.
        https://github.com/langchain-ai/langgraph/blob/main/docs/docs/reference/pregel.md
        """
        try:
            state = await cls._graph.aget_state(config=cls._llm._config)
            interrupts = [
                interrupt for task in state.tasks for interrupt in task.interrupts
            ]
            assistant = next(
                (
                    interrupt
                    for interrupt in reversed(interrupts)
                    if isinstance(interrupt.value, str)
                ),
                None,
            )
            return assistant
        except HTTPStatusError:
            return None
        except (TypeError, AttributeError, KeyError) as e:
            # Handle the case where state or checkpoint is None
            logger.warning(f"Error getting interrupt state: {e}")
            return None

    def _to_message(cls, msg: llm.ChatMessage) -> HumanMessage:
        # Helper used for converting LiveKit inbound to HumanMessage
        if isinstance(msg.content, str):
            content = msg.content
        elif isinstance(msg.content, list):
            content = []
            for c in msg.content:
                if isinstance(c, str):
                    content.append({"type": "text", "text": c})
                elif isinstance(c, llm.ChatImage):
                    if isinstance(c.image, str):
                        content.append({"type": "image_url", "image_url": c.image})
                    else:
                        logger.warning("Unsupported image type")
                else:
                    logger.warning("Unsupported content type")
        else:
            content = ""

        return HumanMessage(content=content, id=msg.id)

    @staticmethod
    def _create_livekit_chunk(
        content: str,
        *,
        id: str | None = None,
    ) -> llm.ChatChunk | None:
        # ChoiceDelta.content must be a string
        return llm.ChatChunk(
            id=id or shortuuid(),
            delta=llm.ChoiceDelta(role="assistant", content=content),
        )

    @staticmethod
    async def _to_livekit_chunk(
        msg: BaseMessageChunk | str | None,
    ) -> llm.ChatChunk | None:
        """Normalize LangGraph message chunk or string into a ChatChunk.

        Accepts:
          - str content
          - message-like objects with .content (str)
          - dicts with {id?, content?}
          - lists where first element carries the content
        Returns None when content is missing or not a string.
        """
        if not msg:
            return None

        request_id = None
        content = msg

        if isinstance(msg, str):
            content = msg
        elif hasattr(msg, "content") and isinstance(msg.content, str):
            request_id = getattr(msg, "id", None)
            content = msg.content
        elif isinstance(msg, dict):
            request_id = msg.get("id")
            content = msg.get("content")
        elif isinstance(msg, list):
            # Handle case where msg is a list - try to extract content from first item
            if msg and len(msg) > 0:
                first_item = msg[0]
                if isinstance(first_item, str):
                    content = first_item
                elif hasattr(first_item, "content") and isinstance(first_item.content, str):
                    content = first_item.content
                    request_id = getattr(first_item, "id", None)
                elif isinstance(first_item, dict):
                    content = first_item.get("content", "")
                    request_id = first_item.get("id")
                else:
                    logger.warning(f"Unsupported message type in list: {type(first_item)}")
                    return None
            else:
                logger.warning("Empty message list received")
                return None
        else:
            logger.warning(f"Unsupported message type: {type(msg)}")
            return None

        # Ensure content is a string
        if not isinstance(content, str):
            logger.warning(f"Content is not a string: {type(content)}")
            return None

        return LangGraphStream._create_livekit_chunk(content, id=request_id)


class LangGraphAdapter(llm.LLM):
    """Adapter that exposes a LangGraph agent as a LiveKit LLM.

    chat() creates a LangGraphStream that maps ChatContext + tools into
    the agent execution. Tools are passed through so LiveKit can advertise
    capabilities to the calling side when applicable.

    See LiveKit LLM.chat signature and LLMStream contract in the docs.
    """

    def __init__(self, graph: Any, config: dict[str, Any] | None = None):
        super().__init__()
        self._graph = graph
        self._config = config

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[FunctionTool | RawFunctionTool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> llm.LLMStream:
        """Create a streaming session backed by the provided LangGraph.

        - chat_ctx: prior conversation context from LiveKit
        - tools: tool definitions (forwarded to base stream for metadata)
        - conn_options: stream connection options
        """
        return LangGraphStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            graph=self._graph,
        )