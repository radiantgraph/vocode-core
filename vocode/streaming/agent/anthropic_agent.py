import json
import os
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import sentry_sdk
from anthropic import AsyncAnthropic, AsyncStream
from anthropic.types import ContentBlockDelta, ContentBlockStart, MessageStreamEvent
from loguru import logger

from vocode.streaming.action.abstract_factory import AbstractActionFactory
from vocode.streaming.action.default_factory import DefaultActionFactory
from vocode.streaming.agent.anthropic_utils import (
    extract_tool_call_data,
    format_anthropic_chat_messages_from_transcript,
    format_tool_result_for_anthropic,
)
from vocode.streaming.agent.base_agent import GeneratedResponse, RespondAgent, StreamedResponse
from vocode.streaming.agent.streaming_utils import collate_response_async, stream_response_async
from vocode.streaming.models.actions import FunctionFragment
from vocode.streaming.models.agent import AnthropicAgentConfig
from vocode.streaming.models.message import BaseMessage, LLMToken, ToolUseMessage, ToolResultMessage
from vocode.streaming.vector_db.factory import VectorDBFactory
from vocode.utils.sentry_utils import CustomSentrySpans, sentry_create_span


class AnthropicAgent(RespondAgent[AnthropicAgentConfig]):
    anthropic_client: AsyncAnthropic

    def __init__(
        self,
        agent_config: AnthropicAgentConfig,
        action_factory: AbstractActionFactory = DefaultActionFactory(),
        vector_db_factory=VectorDBFactory(),
        **kwargs,
    ):
        super().__init__(
            agent_config=agent_config,
            action_factory=action_factory,
            **kwargs,
        )
        self.anthropic_client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        # Track active tool calls
        self._active_tool_calls: Dict[str, Dict[str, Any]] = {}
        # Store pending tool responses for next messages
        self._pending_tool_responses: List[Dict[str, Any]] = []

    def get_chat_parameters(self, messages: list = [], use_functions: bool = True):
        assert self.transcript is not None

        parameters: dict[str, Any] = {
            "messages": messages,
            "system": self.agent_config.prompt_preamble,
            "max_tokens": self.agent_config.max_tokens,
            "temperature": self.agent_config.temperature,
            "stream": True,
        }

        parameters["model"] = self.agent_config.model_name

        # Add tools if configured
        if self.agent_config.tools and use_functions:
            parameters["tools"] = self.agent_config.tools
            
        # Add tool_choice if configured
        if self.agent_config.tool_choice and use_functions:
            parameters["tool_choice"] = self.agent_config.tool_choice

        return parameters

    async def token_generator(
        self,
        gen: AsyncStream[MessageStreamEvent],
    ) -> AsyncGenerator[Union[str, FunctionFragment, ToolUseMessage], None]:
        # Track current tool call construction
        current_tool_id = None
        current_tool_name = None
        current_tool_input = ""
        tool_call_in_progress = False

        async for chunk in gen:
            # Handle text content
            if chunk.type == "content_block_delta" and chunk.delta.type == "text_delta":
                yield chunk.delta.text
                
            # Handle tool use start
            elif chunk.type == "content_block_start" and chunk.content_block.type == "tool_use":
                tool_call_in_progress = True
                current_tool_id = chunk.content_block.id
                current_tool_name = chunk.content_block.name
                self._active_tool_calls[current_tool_id] = {
                    "tool_id": current_tool_id,
                    "tool_name": current_tool_name,
                    "tool_input": {}
                }
                
            # Handle tool use deltas
            elif (chunk.type == "content_block_delta" and 
                  chunk.delta.type == "tool_use_delta" and 
                  tool_call_in_progress):
                
                if current_tool_id and chunk.delta.input and isinstance(chunk.delta.input, dict):
                    # Update the stored tool call with the new input
                    self._active_tool_calls[current_tool_id]["tool_input"].update(chunk.delta.input)
                    
            # Handle tool use finish
            elif chunk.type == "content_block_stop" and tool_call_in_progress and current_tool_id:
                tool_call_in_progress = False
                
                # Get the completed tool call data
                tool_data = self._active_tool_calls.get(current_tool_id, {})
                
                if tool_data and current_tool_name:
                    # Create a full tool use message
                    tool_message = ToolUseMessage(
                        text=f"Using tool: {current_tool_name}",
                        tool_id=current_tool_id,
                        tool_name=current_tool_name,
                        tool_input=tool_data.get("tool_input", {})
                    )
                    
                    # Create a function fragment for backward compatibility
                    function_fragment = FunctionFragment(
                        name=current_tool_name,
                        arguments=json.dumps(tool_data.get("tool_input", {})),
                        tool_id=current_tool_id,
                        is_tool_call=True
                    )
                    
                    # Reset tracking variables
                    current_tool_id = None
                    current_tool_name = None
                    current_tool_input = ""
                    
                    # Yield the tool use message
                    yield tool_message

    async def _get_anthropic_stream(self, chat_parameters: Dict[str, Any]):
        return await self.anthropic_client.messages.create(**chat_parameters)

    async def add_tool_result(
        self,
        tool_id: str,
        tool_name: str,
        result: Dict[str, Any],
        status: str = "success"
    ) -> None:
        """Add a tool result to be included in the next message to Claude"""
        self._pending_tool_responses.append({
            "tool_id": tool_id,
            "tool_name": tool_name,
            "result": result,
            "status": status
        })

    async def _include_pending_tool_results(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Include any pending tool results in the message list"""
        result_messages = messages.copy()
        
        # Add tool results as user messages
        for tool_response in self._pending_tool_responses:
            result_messages.append({
                "role": "user",
                "content": [format_tool_result_for_anthropic(
                    tool_id=tool_response["tool_id"],
                    tool_name=tool_response["tool_name"],
                    result=tool_response["result"],
                    status=tool_response["status"]
                )]
            })
        
        # Clear pending responses after adding them
        self._pending_tool_responses = []
        
        return result_messages

    async def generate_response(
        self,
        human_input,
        conversation_id: str,
        is_interrupt: bool = False,
        bot_was_in_medias_res: bool = False,
    ) -> AsyncGenerator[GeneratedResponse, None]:
        if not self.transcript:
            raise ValueError("A transcript is not attached to the agent")
        messages = format_anthropic_chat_messages_from_transcript(transcript=self.transcript)
        
        # Include any pending tool results in the messages
        if self._pending_tool_responses:
            messages = await self._include_pending_tool_results(messages)
            
        chat_parameters = self.get_chat_parameters(messages)
        try:
            first_sentence_total_span = sentry_create_span(
                sentry_callable=sentry_sdk.start_span, op=CustomSentrySpans.LLM_FIRST_SENTENCE_TOTAL
            )

            ttft_span = sentry_create_span(
                sentry_callable=sentry_sdk.start_span, op=CustomSentrySpans.TIME_TO_FIRST_TOKEN
            )
            stream = await self._get_anthropic_stream(chat_parameters)
        except Exception as e:
            logger.error(
                f"Error while hitting Anthropic with chat_parameters: {chat_parameters}",
                exc_info=True,
            )
            raise e

        response_generator = collate_response_async

        using_input_streaming_synthesizer = (
            self.conversation_state_manager.using_input_streaming_synthesizer()
        )
        if using_input_streaming_synthesizer:
            response_generator = stream_response_async
        async for message in response_generator(
            conversation_id=conversation_id,
            gen=self.token_generator(
                stream,
            ),
            sentry_span=ttft_span,
        ):
            if first_sentence_total_span:
                first_sentence_total_span.finish()

            ResponseClass = (
                StreamedResponse if using_input_streaming_synthesizer else GeneratedResponse
            )
            MessageType = LLMToken if using_input_streaming_synthesizer else BaseMessage

            if isinstance(message, str):
                yield ResponseClass(
                    message=MessageType(text=message),
                    is_interruptible=True,
                )
            elif isinstance(message, ToolUseMessage):
                # Handle tool use message
                yield ResponseClass(
                    message=message,
                    is_interruptible=False,  # Tool calls should not be interrupted
                )
                
                # Process the tool call here or delegate to action system
                # This is where you would implement the actual tool execution logic
                # For now, we just log the tool call
                logger.info(f"Tool call received: {message.tool_name} with inputs: {message.tool_input}")
            else:
                yield ResponseClass(
                    message=message,
                    is_interruptible=True,
                )
