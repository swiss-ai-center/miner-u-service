import asyncio
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from common_code.config import get_settings
from common_code.http_client import HttpClient
from common_code.logger.logger import get_logger, Logger
from common_code.service.controller import router as service_router
from common_code.service.service import ServiceService
from common_code.storage.service import StorageService
from common_code.tasks.controller import router as tasks_router
from common_code.tasks.service import TasksService
from common_code.tasks.models import TaskData
from common_code.service.models import Service
from common_code.service.enums import ServiceStatus
from common_code.common.enums import FieldDescriptionType, ExecutionUnitTagName, ExecutionUnitTagAcronym
from common_code.common.models import FieldDescription, ExecutionUnitTag
from contextlib import asynccontextmanager

# Imports required by the service's model
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
from mineru_vl_utils import MinerUClient
import io
import json

settings = get_settings()


class MyService(Service):
    """
    MinerU service model
    """

    # Any additional fields must be excluded for Pydantic to work
    _model: object
    _logger: Logger
    _processor: AutoProcessor
    _client: MinerUClient

    def __init__(self):
        super().__init__(
            name="MinerU",
            slug="miner-u-service",
            url=settings.service_url,
            summary=api_summary,
            description=api_description,
            status=ServiceStatus.AVAILABLE,
            data_in_fields=[
                FieldDescription(
                    name="image",
                    type=[
                        FieldDescriptionType.IMAGE_PNG,
                        FieldDescriptionType.IMAGE_JPEG,
                    ],
                ),
            ],
            data_out_fields=[
                FieldDescription(
                    name="result", type=[FieldDescriptionType.APPLICATION_JSON]
                ),
            ],
            tags=[
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.IMAGE_PROCESSING,
                    acronym=ExecutionUnitTagAcronym.IMAGE_PROCESSING,
                ),
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.DOCUMENT_PROCESSING,
                    acronym=ExecutionUnitTagAcronym.DOCUMENT_PROCESSING,
                ),
            ],
            has_ai=True,
            # OPTIONAL: CHANGE THE DOCS URL TO YOUR SERVICE'S DOCS
            docs_url="https://docs.swiss-ai-center.ch/reference/services/miner-u/",
        )
        self._logger = get_logger(settings)
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            "opendatalab/MinerU2.5-2509-1.2B",
            dtype="auto",  # use `torch_dtype` instead of `dtype` for transformers<4.56.0
            device_map="auto"
        )
        self._processor = AutoProcessor.from_pretrained("opendatalab/MinerU2.5-2509-1.2B", use_fast=True)
        self._client = MinerUClient(backend="transformers", model=self._model, processor=self._processor)

    def process(self, data):
        raw = data["image"].data
        image = Image.open(io.BytesIO(raw))

        extracted_blocks = self._client.two_step_extract(image)
        formatted_extracted_blocks = []
        for block in extracted_blocks:
            bbox = block.get("bbox", None)
            if bbox:
                formatted_block = {
                    "type": block.get("type", ""),
                    "text": block.get("content", ""),
                    "position": {
                        "left": bbox[0] * image.width,
                        "top": bbox[1] * image.height,
                        "width": (bbox[2] - bbox[0]) * image.width,
                        "height": (bbox[3] - bbox[1]) * image.height,
                    }
                }
            else:
                formatted_block = {
                    "type": block.get("type", ""),
                    "text": block.get("content", ""),
                    "position": None
                }
            formatted_extracted_blocks.append(formatted_block)

        result = {
            "boxes": formatted_extracted_blocks
        }

        return {
            "result": TaskData(data=json.dumps(result).encode("utf-8"),
                               type=FieldDescriptionType.APPLICATION_JSON)
        }


service_service: ServiceService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Manual instances because startup events doesn't support Dependency Injection
    # https://github.com/tiangolo/fastapi/issues/2057
    # https://github.com/tiangolo/fastapi/issues/425

    # Global variable
    global service_service

    # Startup
    logger = get_logger(settings)
    http_client = HttpClient()
    storage_service = StorageService(logger)
    my_service = MyService()
    tasks_service = TasksService(logger, settings, http_client, storage_service)
    service_service = ServiceService(logger, settings, http_client, tasks_service)

    tasks_service.set_service(my_service)

    # Start the tasks service
    tasks_service.start()

    async def announce():
        retries = settings.engine_announce_retries
        for engine_url in settings.engine_urls:
            announced = False
            while not announced and retries > 0:
                announced = await service_service.announce_service(my_service, engine_url)
                retries -= 1
                if not announced:
                    time.sleep(settings.engine_announce_retry_delay)
                    if retries == 0:
                        logger.warning(
                            f"Aborting service announcement after "
                            f"{settings.engine_announce_retries} retries"
                        )

    # Announce the service to its engine
    asyncio.ensure_future(announce())

    yield

    # Shutdown
    for engine_url in settings.engine_urls:
        await service_service.graceful_shutdown(my_service, engine_url)


api_description = """
The service uses MinerU model to extract structured information from images of documents.
"""
api_summary = """
Extract structured information from document images using MinerU.
"""

# Define the FastAPI application with information
app = FastAPI(
    lifespan=lifespan,
    title="MinerU API.",
    description=api_description,
    version="1.0.0",
    contact={
        "name": "Swiss AI Center",
        "url": "https://swiss-ai-center.ch/",
        "email": "info@swiss-ai-center.ch",
    },
    swagger_ui_parameters={
        "tagsSorter": "alpha",
        "operationsSorter": "method",
    },
    license_info={
        "name": "GNU Affero General Public License v3.0 (GNU AGPLv3)",
        "url": "https://choosealicense.com/licenses/agpl-3.0/",
    },
)

# Include routers from other files
app.include_router(service_router, tags=["Service"])
app.include_router(tasks_router, tags=["Tasks"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Redirect to docs
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/docs", status_code=301)
