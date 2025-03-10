import abc
import asyncio
from functools import partial
from typing import List, Optional

import httpx
from fastapi import APIRouter, Form, Request, Response
from loguru import logger
from pydantic.v1 import BaseModel, Field

from vocode.streaming.agent.abstract_factory import AbstractAgentFactory
from vocode.streaming.agent.default_factory import DefaultAgentFactory
from vocode.streaming.models.agent import AgentConfig
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
from vocode.streaming.telephony.conversation.outbound_call import OutboundCall  # Corrected import
from vocode.streaming.telephony.server.router.calls import CallsRouter
from vocode.streaming.telephony.templater import get_connection_twiml
from vocode.streaming.transcriber.abstract_factory import AbstractTranscriberFactory
from vocode.streaming.transcriber.default_factory import DefaultTranscriberFactory
from vocode.streaming.utils import create_conversation_id
from vocode.streaming.utils.events_manager import EventsManager


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
    ):
        self.base_url = base_url
        self.router = APIRouter()
        self.config_manager = config_manager
        self.events_manager = events_manager
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

        # # Register the /make_call endpoint using partial to bind 'self'
        # self.router.add_api_route(
        #     "/make_call",
        #     partial(self.make_call),  # Bind 'self' using partial
        #     methods=["POST"],
        # )
        # logger.info(
        #     f"Set up make_call endpoint at https://{self.base_url}/make_call"
        # )

    def events(self, request: Request):
        return Response()

    async def recordings(self, request: Request, conversation_id: str):
        recording_url = (await request.json())["recording_url"]
        if self.events_manager is not None and recording_url is not None:
            self.events_manager.publish_event(
                RecordingEvent(recording_url=recording_url, conversation_id=conversation_id)
            )
        return Response()

    # async def make_call(
    #     self,
    #     request: Request,
    #     background_tasks: BackgroundTasks
    # ):
    #     try:
    #         data = await request.json()
    #         to_phone = data["to_phone"]
    #         flag = data["flag"]
    #     except (ValueError, KeyError) as e:
    #         logger.error(f"Invalid request body: {e}")
    #         return Response(status_code=400, content="Invalid request body. 'to_phone' and 'flag' are required.")

    #     # Retrieve the corresponding agent_config based on the flag
    #     agent_config = self.outbound_call_configs.agent_configs.get(flag)
    #     if not agent_config:
    #         logger.error(f"Invalid or missing flag provided: {flag}")
    #         return Response(status_code=400, content="Invalid or missing flag provided.")

    #     # Create an instance of OutboundCall with the retrieved agent_config
    #     outbound_call = OutboundCall(
    #         base_url=self.base_url,
    #         to_phone=to_phone,
    #         from_phone=os.environ["TWILIO_FROM_PHONE"],  # Ensure this env variable is set
    #         config_manager=self.config_manager,
    #         agent_config=agent_config,
    #         telephony_config=TwilioConfig(
    #             account_sid=os.environ["TWILIO_ACCOUNT_SID"],
    #             auth_token=os.environ["TWILIO_AUTH_TOKEN"],
    #         ),
    #         synthesizer_config=self.outbound_synthesizer_config,
    #     )

    #     # Start the outbound call in the background
    #     background_tasks.add_task(outbound_call.start)

    #     logger.info(f"Outbound call initiated to {to_phone} with flag {flag}")

    #     return {"status": "Outbound call initiated."}

    async def start_twilio_inbound_recording(
        self, twilio_sid: str, auth_token: str, conversation_id: str
    ) -> None:
        url: str = (
            f"https://api.twilio.com/2010-04-01/Accounts/{twilio_sid}/Calls/{conversation_id}/Recordings.json"
        )

        async with httpx.AsyncClient() as client:
            response: httpx.Response = await client.post(
                url, data={}, auth=(twilio_sid, auth_token)
            )

        if response.status_code != 201:
            logger.warning(f"Failed to start recording: {response.text}")
        else:
            logger.info(f"Recording started for call {twilio_sid}")

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

            logger.info(f"agent is ready to pick up call from {twilio_from}")

            conversation_id = create_conversation_id()
            await self.config_manager.save_config(conversation_id, call_config)

            async def delay_start_recording() -> None:
                await asyncio.sleep(1)
                await self.start_twilio_inbound_recording(
                    twilio_sid=twilio_config.account_sid,
                    auth_token=twilio_config.auth_token,
                    conversation_id=twilio_sid,
                )

            if twilio_config.record:
                asyncio.create_task(coro=delay_start_recording())

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

    def get_router(self) -> APIRouter:
        return self.router
