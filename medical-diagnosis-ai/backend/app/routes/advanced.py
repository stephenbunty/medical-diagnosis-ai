"""Optional features: LLM chat assistant, model metadata."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.config import get_settings
from app.deps import get_current_user
from app.models.user import User
from app.services import report_service

router = APIRouter(prefix="/assistant", tags=["Advanced"])
settings = get_settings()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    context: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    reply: str
    source: str = "rules"


@router.post("/chat", response_model=ChatResponse)
async def medical_chat(
    body: ChatRequest,
    user: Annotated[User, Depends(get_current_user)],
) -> ChatResponse:
    """
    Lightweight assistant: uses OpenAI when `OPENAI_API_KEY` is set, otherwise safe templated guidance.
    Not a substitute for clinical decision support validation.
    """
    _ = user
    if settings.openai_api_key:
        text = await report_service.maybe_llm_explanation(
            {"user_message": body.message, "context": body.context or {}}
        )
        if text:
            return ChatResponse(reply=text, source="llm")
    safe = (
        "This assistant cannot provide diagnoses. "
        "Review the structured report, heatmaps, and source images; "
        "correlate with history and labs; involve radiology as needed."
    )
    return ChatResponse(reply=f"{safe}\n\nYou asked: {body.message[:500]}", source="rules")


@router.get("/model-info")
async def model_info(user: Annotated[User, Depends(get_current_user)]) -> dict[str, Any]:
    _ = user
    return {
        "chest": "Multi-label ResNet50 (6 conditions) + Grad-CAM on layer4",
        "brain": "4-class ResNet18 + Grad-CAM",
        "segmentation": "U-Net 256x256 (binary mask) with saliency fallback if no weights",
        "framework": "PyTorch + Torchvision",
    }
