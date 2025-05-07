from enum import Enum
from typing import Any, Dict, List, Optional

from .model import TypedModel


class MessageType(str, Enum):
    BASE = "message_base"
    SSML = "message_ssml"
    BOT_BACKCHANNEL = "bot_backchannel"
    LLM_TOKEN = "llm_token"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"


class BaseMessage(TypedModel, type=MessageType.BASE):  # type: ignore
    text: str
    trailing_silence_seconds: float = 0.0
    cache_phrase: Optional[str] = None


class SSMLMessage(BaseMessage, type=MessageType.SSML):  # type: ignore
    ssml: str


class BotBackchannel(BaseMessage, type=MessageType.BOT_BACKCHANNEL):  # type: ignore
    pass


class LLMToken(BaseMessage, type=MessageType.LLM_TOKEN):  # type: ignore
    pass


class ToolUseMessage(BaseMessage, type=MessageType.TOOL_USE):  # type: ignore
    tool_id: str
    tool_name: str
    tool_input: Dict[str, Any]


class ToolResultMessage(BaseMessage, type=MessageType.TOOL_RESULT):  # type: ignore
    tool_id: str
    tool_name: str
    tool_result: Dict[str, Any]
    status: str = "success"  # can be "success" or "error"

class SilenceMessage(BotBackchannel):
    text: str = "<silence>"
    trailing_silence_seconds: float = 1.0
