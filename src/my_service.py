from common_code.config import get_settings
from common_code.logger.logger import get_logger, Logger
from common_code.service.models import Service
from common_code.service.enums import ServiceStatus
from common_code.common.enums import FieldDescriptionType, ExecutionUnitTagName, ExecutionUnitTagAcronym
from common_code.common.models import FieldDescription, ExecutionUnitTag
from common_code.tasks.models import TaskData

# Imports required by the service's model
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
from mineru_vl_utils import MinerUClient
import io
import json

settings = get_settings()

api_description = """
The service uses MinerU model to extract structured information from images of documents.
"""
api_summary = """
Extract structured information from document images using MinerU.
"""
api_title = "MinerU API."
version = "1.0.0"


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
