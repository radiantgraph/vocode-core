import abc
from functools import partial
from typing import List, Optional

from fastapi import APIRouter, Form, Request, Response, BackgroundTasks
from loguru import logger
from pydantic.v1 import BaseModel, Field

from vocode.streaming.agent.abstract_factory import AbstractAgentFactory
from vocode.streaming.agent.default_factory import DefaultAgentFactory
from vocode.streaming.models.agent import AgentConfig, ChatGPTAgentConfig
from vocode.streaming.models.events import RecordingEvent
from vocode.streaming.models.synthesizer import SynthesizerConfig  
from vocode.streaming.models.telephony import (
    TwilioCallConfig,
    TwilioConfig,
    VonageCallConfig,
    VonageConfig,
)
from vocode.streaming.models.transcriber import TranscriberConfig
from vocode.streaming.synthesizer.abstract_factory import AbstractSynthesizerFactory
from vocode.streaming.synthesizer.default_factory import DefaultSynthesizerFactory
from vocode.streaming.telephony.client.abstract_telephony_client import AbstractTelephonyClient
from vocode.streaming.telephony.client.twilio_client import TwilioClient
from vocode.streaming.telephony.client.vonage_client import VonageClient
from vocode.streaming.telephony.config_manager.base_config_manager import BaseConfigManager
from vocode.streaming.telephony.server.router.calls import CallsRouter
from vocode.streaming.telephony.templater import get_connection_twiml
from vocode.streaming.transcriber.abstract_factory import AbstractTranscriberFactory
from vocode.streaming.transcriber.default_factory import DefaultTranscriberFactory
from vocode.streaming.utils import create_conversation_id
from vocode.streaming.utils.events_manager import EventsManager
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.telephony.conversation.outbound_call import OutboundCall  # Corrected import
import os


class AbstractInboundCallConfig(BaseModel, abc.ABC):
    url: str
    agent_config: AgentConfig
    transcriber_config: Optional[TranscriberConfig] = None
    synthesizer_config: Optional[SynthesizerConfig] = None


class TwilioInboundCallConfig(AbstractInboundCallConfig):
    twilio_config: TwilioConfig


class VonageInboundCallConfig(AbstractInboundCallConfig):
    vonage_config: VonageConfig


class VonageAnswerRequest(BaseModel):
    to: str
    from_: str = Field(..., alias="from")
    uuid: str


class MakeCallRequest(BaseModel):
    to_phone: str
    flag: str  # Should be one of 'demo1', 'demo2', 'demo3'


class TelephonyServer:
    def __init__(
        self,
        base_url: str,
        config_manager: BaseConfigManager,
        inbound_call_configs: List[AbstractInboundCallConfig] = [],
        transcriber_factory: AbstractTranscriberFactory = DefaultTranscriberFactory(),
        agent_factory: AbstractAgentFactory = DefaultAgentFactory(),
        synthesizer_factory: AbstractSynthesizerFactory = DefaultSynthesizerFactory(),
        events_manager: Optional[EventsManager] = None,
        outbound_synthesizer_config: Optional[SynthesizerConfig] = None,  # {{ edit_1 }} Add this parameter
    ):
        self.base_url = base_url
        self.router = APIRouter()
        self.config_manager = config_manager
        self.events_manager = events_manager
        self.outbound_synthesizer_config = outbound_synthesizer_config  # {{ edit_2 }} Store the outbound synthesizer config

        self.router.include_router(
            CallsRouter(
                base_url=base_url,
                config_manager=self.config_manager,
                transcriber_factory=transcriber_factory,
                agent_factory=agent_factory,
                synthesizer_factory=synthesizer_factory,
                events_manager=self.events_manager,
            ).get_router()
        )
        for config in inbound_call_configs:
            self.router.add_api_route(
                config.url,
                self.create_inbound_route(inbound_call_config=config),
                methods=["POST"],
            )
        # vonage requires an events endpoint
        self.router.add_api_route("/events", self.events, methods=["GET", "POST"])
        logger.info(f"Set up events endpoint at https://{self.base_url}/events")

        self.router.add_api_route(
            "/recordings/{conversation_id}", self.recordings, methods=["GET", "POST"]
        )
        logger.info(
            f"Set up recordings endpoint at https://{self.base_url}/recordings/{{conversation_id}}"
        )

        # Add the new /make_call API route
        self.router.add_api_route(
            "/make_call",
            self.make_call,
            methods=["POST"],
        )
        logger.info(
            f"Set up make_call endpoint at https://{self.base_url}/make_call"
        )

    def events(self, request: Request):
        return Response()

    async def recordings(self, request: Request, conversation_id: str):
        recording_url = (await request.json())["recording_url"]
        if self.events_manager is not None and recording_url is not None:
            self.events_manager.publish_event(
                RecordingEvent(recording_url=recording_url, conversation_id=conversation_id)
            )
        return Response()

    def create_inbound_route(
        self,
        inbound_call_config: AbstractInboundCallConfig,
    ):
        async def twilio_route(
            twilio_config: TwilioConfig,
            twilio_sid: str = Form(alias="CallSid"),
            twilio_from: str = Form(alias="From"),
            twilio_to: str = Form(alias="To"),
        ) -> Response:
            call_config = TwilioCallConfig(
                transcriber_config=inbound_call_config.transcriber_config
                or TwilioCallConfig.default_transcriber_config(),
                agent_config=inbound_call_config.agent_config,
                synthesizer_config=inbound_call_config.synthesizer_config
                or TwilioCallConfig.default_synthesizer_config(),
                twilio_config=twilio_config,
                twilio_sid=twilio_sid,
                from_phone=twilio_from,
                to_phone=twilio_to,
                direction="inbound",
            )

            conversation_id = create_conversation_id()
            await self.config_manager.save_config(conversation_id, call_config)
            return get_connection_twiml(base_url=self.base_url, call_id=conversation_id)

        async def vonage_route(vonage_config: VonageConfig, request: Request):
            vonage_answer_request = VonageAnswerRequest.parse_obj(await request.json())
            call_config = VonageCallConfig(
                transcriber_config=inbound_call_config.transcriber_config
                or VonageCallConfig.default_transcriber_config(),
                agent_config=inbound_call_config.agent_config,
                synthesizer_config=inbound_call_config.synthesizer_config
                or VonageCallConfig.default_synthesizer_config(),
                vonage_config=vonage_config,
                vonage_uuid=vonage_answer_request.uuid,
                to_phone=vonage_answer_request.from_,
                from_phone=vonage_answer_request.to,
                direction="inbound",
            )
            conversation_id = create_conversation_id()
            await self.config_manager.save_config(conversation_id, call_config)
            vonage_client = VonageClient(
                base_url=self.base_url,
                maybe_vonage_config=vonage_config,
                record_calls=vonage_config.record,
            )
            return vonage_client.create_call_ncco(
                conversation_id=conversation_id,
                record=vonage_config.record,
            )

        if isinstance(inbound_call_config, TwilioInboundCallConfig):
            logger.info(
                f"Set up inbound call TwiML at https://{self.base_url}{inbound_call_config.url}"
            )
            return partial(twilio_route, inbound_call_config.twilio_config)
        elif isinstance(inbound_call_config, VonageInboundCallConfig):
            logger.info(
                f"Set up inbound call NCCO at https://{self.base_url}{inbound_call_config.url}"
            )
            return partial(vonage_route, inbound_call_config.vonage_config)
        else:
            raise ValueError(f"Unknown inbound call config type {type(inbound_call_config)}")

    async def end_outbound_call(self, conversation_id: str):
        # TODO validation via twilio_client
        call_config = await self.config_manager.get_config(conversation_id)
        if not call_config:
            raise ValueError(f"Could not find call config for {conversation_id}")
        telephony_client: AbstractTelephonyClient
        if isinstance(call_config, TwilioCallConfig):
            telephony_client = TwilioClient(
                base_url=self.base_url, maybe_twilio_config=call_config.twilio_config
            )
            await telephony_client.end_call(call_config.twilio_sid)
        elif isinstance(call_config, VonageCallConfig):
            telephony_client = VonageClient(
                base_url=self.base_url, maybe_vonage_config=call_config.vonage_config
            )
            await telephony_client.end_call(call_config.vonage_uuid)
        return {"id": conversation_id}

    # Define the handler for the outbound call
    async def make_call(self, background_tasks: BackgroundTasks):
        make_call_request = MakeCallRequest.parse_obj(await request.json())
        to_phone = make_call_request.to_phone
        flag = make_call_request.flag

        # Validate the flag
        if flag not in {"demo1", "demo2", "demo3"}:
            return Response(status_code=400, content="Invalid flag provided.")

        # Set up agent_config based on the flag
        if flag == "demo1":
            agent_config = ChatGPTAgentConfig(
                initial_message=BaseMessage(text="Demo1 initial message"),
                prompt_preamble="Demo1 prompt preamble",
                generate_responses=True,
                model_name="gpt-4o"
            )
        elif flag == "demo2":
            agent_config = ChatGPTAgentConfig(
                initial_message=BaseMessage(text="Demo2 initial message"),
                prompt_preamble="Demo2 prompt preamble",
                generate_responses=True,
                model_name="gpt-4o"
            )
        elif flag == "demo3":
            agent_config = ChatGPTAgentConfig(
                initial_message=BaseMessage(text="Demo3 initial message"),
                prompt_preamble="Demo3 prompt preamble",
                generate_responses=True,
                model_name="gpt-4o"
            )

        # Create an instance of OutboundCall
        outbound_call = OutboundCall(
            base_url=self.base_url,
            to_phone=to_phone,
            from_phone=os.environ["TWILIO_FROM_PHONE"],  # Ensure this env variable is set
            config_manager=self.config_manager,
            agent_config=agent_config,
            telephony_config=TwilioConfig(
                account_sid=os.environ["TWILIO_ACCOUNT_SID"],
                auth_token=os.environ["TWILIO_AUTH_TOKEN"],
            ),
            synthesizer_config=self.outbound_synthesizer_config,  # {{ edit_3 }} Pass the synthesizer_config
        )

        # Start the outbound call in the background
        background_tasks.add_task(outbound_call.start)

        return {"status": "Outbound call initiated."}

    def get_router(self) -> APIRouter:
        return self.router
