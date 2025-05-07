from typing import Any, Dict, List, Optional

from vocode.streaming.agent.openai_utils import merge_event_logs
from vocode.streaming.models.message import ToolResultMessage, ToolUseMessage
from vocode.streaming.models.transcript import ActionStart, EventLog, Transcript, TranscriptEvent


def format_anthropic_chat_messages_from_transcript(
    transcript: Transcript,
) -> list[dict]:
    # merge consecutive bot messages
    new_event_logs: list[EventLog] = merge_event_logs(event_logs=transcript.event_logs)

    # Removing BOT_ACTION_START so that it doesn't confuse the completion-y prompt, e.g.
    # BOT: BOT_ACTION_START: action_end_conversation
    # Right now, this version of context does not work for normal actions, only phrase trigger actions
    merged_event_logs_sans_bot_action_start = [
        event_log for event_log in new_event_logs if not isinstance(event_log, ActionStart)
    ]
    
    # Process the transcript events to include tools
    messages = []
    current_role = "user"
    current_content = []
    tool_events = []
    
    for event in merged_event_logs_sans_bot_action_start:
        event_type = getattr(event, "event_type", None)
        
        # Handle tool use events
        if isinstance(event.message, ToolUseMessage):
            # If we have text content, add it first
            if current_content and current_role == "assistant":
                messages.append({
                    "role": current_role,
                    "content": [{"type": "text", "text": " ".join(current_content)}]
                })
                current_content = []
            
            # Add tool use content
            messages.append({
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": event.message.tool_id,
                    "name": event.message.tool_name,
                    "input": event.message.tool_input
                }]
            })
            tool_events.append(event.message)
            continue
        
        # Handle tool result events
        if isinstance(event.message, ToolResultMessage):
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": event.message.tool_id,
                    "content": event.message.tool_result
                }]
            })
            continue
        
        # Regular message handling
        if event_type == "bot":
            if current_role != "assistant":
                if current_content:
                    messages.append({
                        "role": current_role,
                        "content": Transcript(event_logs=[event]).to_string(
                            include_timestamps=False,
                            mark_human_backchannels_with_brackets=True,
                        ),
                    })
                    current_content = []
                current_role = "assistant"
            current_content.append(event.message.text)
        elif event_type == "human":
            if current_role != "user":
                if current_content:
                    messages.append({
                        "role": current_role,
                        "content": " ".join(current_content),
                    })
                    current_content = []
                current_role = "user"
            current_content.append(event.message.text)
    
    # Add any remaining content
    if current_content:
        content_value = " ".join(current_content)
        if current_role == "user":
            messages.append({
                "role": current_role,
                "content": content_value,
            })
        else:  # assistant
            messages.append({
                "role": current_role,
                "content": content_value,
            })
    
    # If we don't have any messages or the last one isn't the assistant, add a starter message
    if not messages or messages[-1]["role"] != "assistant":
        messages.append({"role": "assistant", "content": "BOT:"})
    
    return messages
    # TODO: reliably count tokens of Anthropic messages so that we don't exceed the context window


def merge_bot_messages_for_langchain(messages: list[tuple]) -> list[tuple]:
    merged_messages: list[tuple] = []
    for role, message in messages:
        if role == "ai" and merged_messages and merged_messages[-1][0] == "ai":
            merged_messages[-1] = ("ai", merged_messages[-1][1] + message)
        else:
            merged_messages.append((role, message))
    return merged_messages


def format_tool_result_for_anthropic(tool_id: str, tool_name: str, result: Dict[str, Any], status: str = "success") -> Dict:
    """Format tool result for Claude's expected structure"""
    return {
        "type": "tool_result",
        "tool_use_id": tool_id,
        "content": result
    }


def extract_tool_call_data(tool_use_event) -> Optional[Dict[str, Any]]:
    """Extract tool call data from a Claude tool_use content block"""
    try:
        return {
            "tool_id": tool_use_event.get("id"),
            "tool_name": tool_use_event.get("name"),
            "tool_input": tool_use_event.get("input", {})
        }
    except (AttributeError, KeyError):
        return None
