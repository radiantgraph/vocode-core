from typing import Literal, Optional, Type, Union
from loguru import logger
from pydantic.v1 import BaseModel, Field

from vocode.streaming.action.phone_call_action import TwilioPhoneConversationAction
from vocode.streaming.models.actions import ActionConfig as VocodeActionConfig
from vocode.streaming.models.actions import ActionInput, ActionOutput
from vocode.streaming.utils.phone_numbers import sanitize_phone_number
from vocode.streaming.utils.state_manager import TwilioPhoneConversationStateManager


class SendMessageRequiredParameters(BaseModel):
    phone_number: str = Field(..., description="The phone number to send the message to")
    message: str = Field(..., description="The content of the message")


SendMessageParameters = Union[SendMessageRequiredParameters]


class SendMessageResponse(BaseModel):
    success: bool


class SendMessageVocodeActionConfig(VocodeActionConfig, type="action_send_message"):  # type: ignore
    phone_number: Optional[str] = Field(None, description="The phone number to send the message to")
    message: Optional[str] = Field(None, description="The text message content")

    def get_phone_number(self, input: ActionInput) -> str:
        logger.info("Getting phone number for sending message")
        if isinstance(input.params, SendMessageRequiredParameters):
            return input.params.phone_number
        else:
            assert self.phone_number, "Phone number must be set"
            return self.phone_number

    def get_message(self, input: ActionInput) -> str:
        logger.info("Getting message content")
        if isinstance(input.params, SendMessageRequiredParameters):
            return input.params.message
        else:
            assert self.message, "Message content must be set"
            return self.message

FUNCTION_DESCRIPTION = "Sends a text message to a phone number."
QUIET = True
IS_INTERRUPTIBLE = True
SHOULD_RESPOND: Literal["always"] = "always"


class TwilioSendMessage(
    TwilioPhoneConversationAction[
        SendMessageVocodeActionConfig, SendMessageParameters, SendMessageResponse
    ]
):
    description: str = FUNCTION_DESCRIPTION
    response_type: Type[SendMessageResponse] = SendMessageResponse
    conversation_state_manager: TwilioPhoneConversationStateManager

    @property
    def parameters_type(self) -> Type[SendMessageParameters]:
        if self.action_config.phone_number and self.action_config.message:
            return SendMessageRequiredParameters
        else:
            return SendMessageRequiredParameters

    def __init__(self, action_config: SendMessageVocodeActionConfig):
        super().__init__(
            action_config,
            quiet=QUIET,
            is_interruptible=False,
            should_respond=SHOULD_RESPOND,
        )

    async def send_message(self, to_phone: str, message: str):
        twilio_client = self.conversation_state_manager.create_twilio_client()

        check_phone = twilio_client.get_telephony_config().phone_number

        logger.info(f"Sending message to {check_phone} with content: {message}")

        # Use Twilio's API to send the message
        try:
            response = twilio_client.messages.create(
                body=message,
                from_=+16157096372, #twilio_client.get_telephony_config().phone_number,
                to=to_phone
            )

            if not response.sid:
                logger.error("Failed to send the message")
                raise Exception("Failed to send message")
            logger.info(f"Message sent successfully with SID: {response.sid}")
            return response
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            raise e

    async def run(
        self, action_input: ActionInput[SendMessageParameters]
    ) -> ActionOutput[SendMessageResponse]:
        logger.info("Running send message action with input: {action_input}")
        phone_number = self.action_config.get_phone_number(action_input)
        sanitized_phone_number = sanitize_phone_number(phone_number)
        message = self.action_config.get_message(action_input)

        logger.info(f"Sanitized phone number: {sanitized_phone_number}")
        logger.info(f"Message content: {message}")

        await self.send_message(sanitized_phone_number, message)

        logger.info("Message sent successfully")
        return ActionOutput(
            action_type=action_input.action_config.type,
            response=SendMessageResponse(success=True),
        )