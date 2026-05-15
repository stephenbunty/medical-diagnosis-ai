"""Schemas for upload, prediction, segmentation, and reports."""

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    study_id: int
    modality: str
    filename: str
    message: str = "Upload successful"


class PredictRequest(BaseModel):
    study_id: int
    modality: str = Field(pattern="^(chest_xray|brain_mri)$")
    run_segmentation: bool = True
    run_heatmap: bool = True


class PredictResponse(BaseModel):
    study_id: int
    modality: str
    labels: dict[str, float]
    top_findings: list[str]
    heatmap_url: str | None = None
    mask_url: str | None = None
    severity_score: float
    prediction_id: int


class ReportRequest(BaseModel):
    prediction_id: int
    use_llm: bool = False


class ReportResponse(BaseModel):
    prediction_id: int
    findings: str
    impression: str
    severity_score: float
    suggested_action: str
    llm_explanation: str | None = None


class SegmentationResponse(BaseModel):
    study_id: int
    mask_url: str


class HistoryItem(BaseModel):
    prediction_id: int
    study_id: int
    modality: str
    created_at: str
    labels: dict[str, float]
    severity_score: float
