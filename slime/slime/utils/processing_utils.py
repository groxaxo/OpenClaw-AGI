import base64
import io
import logging
from typing import Any

from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizerBase, ProcessorMixin

logger = logging.getLogger(__name__)

# Default image patch size for vision-language models
# Note: Qwen3-VL uses 16, Qwen2.5-VL uses 14
# Reference: https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/README.md
DEFAULT_PATCH_SIZE = 16


def load_tokenizer(name_or_path: str, **kwargs):
    return AutoTokenizer.from_pretrained(name_or_path, **kwargs)


def load_processor(name_or_path: str, **kwargs):
    try:
        proc = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except (OSError, ValueError) as e:
        logger.warning(f"Failed to load processor from {name_or_path}: {e}")
        proc = None

    # If HF returned a tokenizer, discard it.
    if isinstance(proc, PreTrainedTokenizerBase) or not isinstance(proc, ProcessorMixin):
        proc = None

    return proc


def process_vision_info(prompt, processor):
    # temporary solution, will write image utils for slime later
    from qwen_vl_utils import process_vision_info

    if hasattr(processor.image_processor, "patch_size"):
        image_patch_size = processor.image_processor.patch_size
    else:
        logger.info(f"Using default patch size: {DEFAULT_PATCH_SIZE}")
        image_patch_size = DEFAULT_PATCH_SIZE
    images, videos = process_vision_info(prompt, image_patch_size=image_patch_size)
    multimodal_inputs = {"images": images, "videos": videos}
    return multimodal_inputs


def encode_image_for_rollout_engine(image) -> str:
    """Load an image from path, ensure RGB, encode as PNG base64 string."""
    buffer = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def flatten_openclaw_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                if item is not None:
                    parts.append(str(item))
                continue

            item_type = str(item.get("type") or "").lower()
            if item_type in {"text", "input_text", "output_text"}:
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif item_type in {"image", "image_url"}:
                parts.append("[image]")
            elif item_type == "video":
                parts.append("[video]")
            elif item_type == "audio":
                parts.append("[audio]")
        return " ".join(part for part in parts if part).strip()
    return str(content) if content is not None else ""


def _build_base64_data_url(data: str, mime_type: str) -> str:
    if data.startswith("data:"):
        return data
    return f"data:{mime_type};base64,{data}"


def _normalize_openclaw_multimodal_block(item: dict[str, Any]) -> dict[str, Any] | None:
    item_type = str(item.get("type") or "").lower()
    if item_type == "image":
        image = item.get("image")
        if isinstance(image, str) and image.strip():
            return {"type": "image", "image": image.strip()}

        source = item.get("source")
        if isinstance(source, dict):
            source_type = str(source.get("type") or "").lower()
            if source_type == "base64":
                data = str(source.get("data") or "").strip()
                if not data:
                    return None
                mime_type = str(source.get("media_type") or source.get("mimeType") or "image/png").strip() or "image/png"
                return {"type": "image", "image": _build_base64_data_url(data, mime_type)}

            url = source.get("url") or source.get("uri")
            if isinstance(url, str) and url.strip():
                return {"type": "image", "image": url.strip()}

        data = item.get("data")
        if isinstance(data, str) and data.strip():
            mime_type = str(item.get("mimeType") or item.get("media_type") or "image/png").strip() or "image/png"
            return {"type": "image", "image": _build_base64_data_url(data.strip(), mime_type)}

        url = item.get("url") or item.get("uri")
        if isinstance(url, str) and url.strip():
            return {"type": "image", "image": url.strip()}
        return None

    if item_type == "image_url":
        image_url = item.get("image_url")
        if isinstance(image_url, dict):
            url = image_url.get("url")
        else:
            url = image_url
        if isinstance(url, str) and url.strip():
            return {"type": "image", "image": url.strip()}
        return None

    if item_type == "video":
        video = item.get("video") or item.get("url") or item.get("uri")
        if isinstance(video, str) and video.strip():
            return {"type": "video", "video": video.strip()}
        return None

    return None


def normalize_openclaw_content_for_template(
    content: Any,
    *,
    preserve_multimodal: bool = False,
) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content) if content is not None else ""
    if not preserve_multimodal:
        return flatten_openclaw_message_content(content)

    normalized: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            continue

        item_type = str(item.get("type") or "").lower()
        if item_type in {"text", "input_text", "output_text"}:
            text = item.get("text")
            if text is not None:
                normalized.append({"type": "text", "text": str(text)})
            continue

        multimodal_block = _normalize_openclaw_multimodal_block(item)
        if multimodal_block is not None:
            normalized.append(multimodal_block)

    return normalized if normalized else flatten_openclaw_message_content(content)


def normalize_openclaw_messages_for_template(
    messages: list[dict[str, Any]],
    *,
    preserve_multimodal: bool = False,
) -> list[dict[str, Any]]:
    normalized_messages: list[dict[str, Any]] = []
    for message in messages:
        normalized = dict(message)
        if normalized.get("role") == "developer":
            normalized["role"] = "system"
        normalized["content"] = normalize_openclaw_content_for_template(
            normalized.get("content"),
            preserve_multimodal=preserve_multimodal,
        )
        normalized_messages.append(normalized)
    return normalized_messages


def _messages_include_multimodal_inputs(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").lower()
            if item_type in {"image", "image_url", "video"}:
                return True
    return False


def prepare_openclaw_chat_template_inputs(
    tokenizer,
    processor,
    messages: list[dict[str, Any]],
    *,
    tools: Any = None,
    add_generation_prompt: bool,
) -> tuple[str, list[int], dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    rendered_text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )

    multimodal_inputs = None
    multimodal_train_inputs = None
    image_data: list[str] = []
    if processor is not None and _messages_include_multimodal_inputs(messages):
        candidate_inputs = process_vision_info(messages, processor)
        has_multimodal_inputs = bool(candidate_inputs.get("images") or candidate_inputs.get("videos"))
        if has_multimodal_inputs:
            multimodal_inputs = candidate_inputs
            processor_output = processor(
                text=rendered_text,
                return_tensors="pt",
                **multimodal_inputs,
            )
            input_ids = processor_output["input_ids"][0]
            if hasattr(input_ids, "tolist"):
                input_ids = input_ids.tolist()
            else:
                input_ids = list(input_ids)
            multimodal_train_inputs = {
                key: value
                for key, value in processor_output.items()
                if key not in {"input_ids", "attention_mask"}
            } or None
            if multimodal_inputs.get("images"):
                image_data = [
                    encode_image_for_rollout_engine(image)
                    for image in multimodal_inputs["images"]
                ]
            return rendered_text, [int(token_id) for token_id in input_ids], multimodal_inputs, multimodal_train_inputs, image_data

    input_ids = tokenizer(rendered_text, add_special_tokens=False)["input_ids"]
    return rendered_text, [int(token_id) for token_id in input_ids], None, None, image_data
