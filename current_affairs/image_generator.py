import base64
import time
from io import BytesIO
from PIL import Image as PILImage
from openai import AzureOpenAI
from azure.storage.blob import BlobServiceClient

from current_affairs.config import (
    CA_GEN_AZURE_OPENAI_API_KEY,
    CA_GEN_AZURE_OPENAI_API_VERSION,
    CA_GEN_AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_IMAGE_DEPLOYMENT,
    AZ_CONN_STR,
    AZ_CONTAINER,
    IMAGE_OUTPUT_DIR,
    IMAGE_TIMEOUT,
    DEFAULT_MAX_RETRIES,
)


def gen_image_prompt(topic: str, summary: str) -> str:
    """Build editorial prompt for UPSC educational infographic."""
    return f"""
{topic}

{summary[:500]}

You're an AI capable of converting textual information into concrete images and your task is to convert the information in this article into symbolic images as you understand them.
Create a clean educational infographic for UPSC aspirants.

Style guidelines:
- Clear typography and concise educational labels are allowed
- Flat infographic or polished editorial infographic style
- Avoid cinematic photorealism
- Avoid fictional or highly detailed geographic maps
- Prefer schematic or symbolic map representations when needed
- Landscape orientation
- Suitable for serious civil services exam preparation
""".strip()


def get_image_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=CA_GEN_AZURE_OPENAI_API_KEY,
        api_version=CA_GEN_AZURE_OPENAI_API_VERSION,
        azure_endpoint=CA_GEN_AZURE_OPENAI_ENDPOINT,
        timeout=IMAGE_TIMEOUT,
        max_retries=DEFAULT_MAX_RETRIES,
    )


def get_blob_service_client() -> BlobServiceClient:
    if not AZ_CONN_STR:
        raise ValueError("Azure Storage connection string is missing")
    return BlobServiceClient.from_connection_string(AZ_CONN_STR)


def generate_and_save_image(article_id: int, topic: str, summary: str) -> str:
    """
    Generate an infographic using Azure OpenAI (gpt-image-2), save it locally,
    and upload it to Azure Blob Storage at thumbnail/{article_id}.jpg.
    Returns the uploaded blob URL.
    """
    t_start = time.time()
    try:
        print(f"[IMAGE] Starting image generation for article {article_id}: {topic}", flush=True)

        image_client = get_image_client()
        prompt = gen_image_prompt(topic, summary)

        t_gen = time.time()
        response = image_client.images.generate(
            model=AZURE_OPENAI_IMAGE_DEPLOYMENT,
            prompt=prompt,
            size="1536x1024",
        )
        gen_duration = round(time.time() - t_gen, 1)

        image_b64 = response.data[0].b64_json
        image_bytes = base64.b64decode(image_b64)

        image = PILImage.open(BytesIO(image_bytes)).convert("RGB")
        image_path = IMAGE_OUTPUT_DIR / f"{article_id}.jpg"

        image.save(
            str(image_path),
            format="JPEG",
            quality=82,
            optimize=True,
        )

        print(f"[IMAGE] Generated image in {gen_duration}s for article {article_id}: {image_path}", flush=True)

        blob_name = f"{article_id}.jpg"
        t_blob = time.time()
        blob_service = get_blob_service_client()
        blob_client = blob_service.get_blob_client(
            container=AZ_CONTAINER,
            blob=blob_name,
        )

        with open(str(image_path), "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        blob_url = blob_client.url
        blob_duration = round(time.time() - t_blob, 1)
        total_img_duration = round(time.time() - t_start, 1)

        print(f"[IMAGE] Azure Blob upload completed in {blob_duration}s (Total image pipeline: {total_img_duration}s): {blob_url}", flush=True)
        return blob_url

    except Exception as e:
        print(f"[IMAGE] Image generation failed for article {article_id} after {round(time.time() - t_start, 1)}s: {e}", flush=True)
        raise e
