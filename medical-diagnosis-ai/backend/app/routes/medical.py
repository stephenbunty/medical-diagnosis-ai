"""
Medical imaging endpoints: upload, predict, heatmap, segmentation, reports, history.
All routes require JWT except where noted (none — all protected).
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import get_settings
from app.database import get_db
from app.deps import get_current_user
from app.models.study import PredictionRecord, Study
from app.models.user import User
from app.schemas.prediction import (
    HistoryItem,
    PredictRequest,
    PredictResponse,
    ReportRequest,
    ReportResponse,
    SegmentationResponse,
    UploadResponse,
)
from app.services import report_service
from app.services.ml_service import ml_service
from app.utils.image_io import load_image_rgb

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Medical AI"])
settings = get_settings()


def _ensure_dirs() -> None:
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.heatmap_dir.mkdir(parents=True, exist_ok=True)
    settings.mask_dir.mkdir(parents=True, exist_ok=True)


def _safe_suffix(name: str) -> str:
    lower = name.lower()
    for ext in (".png", ".jpg", ".jpeg", ".dcm", ".dicom"):
        if lower.endswith(ext):
            return ext
    return ".png"


@router.post("/upload", response_model=UploadResponse)
async def upload(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    file: UploadFile = File(...),
    modality: str = Form(..., pattern="^(chest_xray|brain_mri)$"),
) -> UploadResponse:
    """Upload PNG, JPEG, or DICOM for inference."""
    _ensure_dirs()
    suffix = _safe_suffix(file.filename or "image.png")
    dest = settings.upload_dir / f"user_{user.id}_{file.filename or 'upload'}{suffix}"
    try:
        with dest.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        await file.close()

    study = Study(
        user_id=user.id,
        modality=modality,
        original_filename=file.filename or dest.name,
        stored_path=str(dest.resolve()),
    )
    db.add(study)
    await db.commit()
    await db.refresh(study)
    return UploadResponse(study_id=study.id, modality=modality, filename=study.original_filename)


@router.post("/predict", response_model=PredictResponse)
async def predict(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    body: PredictRequest,
) -> PredictResponse:
    """Run classification + optional Grad-CAM and segmentation."""
    _ensure_dirs()
    result = await db.execute(
        select(Study).where(Study.id == body.study_id, Study.user_id == user.id)
    )
    study = result.scalar_one_or_none()
    if not study:
        raise HTTPException(status_code=404, detail="Study not found")
    if study.modality != body.modality:
        raise HTTPException(status_code=400, detail="Modality mismatch with stored study")

    path = Path(study.stored_path)
    if not path.exists():
        raise HTTPException(status_code=400, detail="File missing on server")

    if body.modality == "chest_xray":
        out = ml_service.predict_chest(path)
    else:
        out = ml_service.predict_brain(path)

    heatmap_rel: str | None = None
    mask_rel: str | None = None

    if body.run_heatmap:
        hm_name = f"hm_{study.id}_{body.modality}.png"
        hm_path = settings.heatmap_dir / hm_name
        if body.modality == "chest_xray":
            ml_service.gradcam_chest(out["tensor"], out["pil"], hm_path)
        else:
            ml_service.gradcam_brain(out["tensor"], out["pil"], hm_path)
        heatmap_rel = hm_name

    if body.run_segmentation:
        mk_name = f"mask_{study.id}_{body.modality}.png"
        mk_path = settings.mask_dir / mk_name
        ml_service.segment(out["pil"], body.modality, mk_path)
        mask_rel = mk_name

    pred = PredictionRecord(
        study_id=study.id,
        labels_json=json.dumps(out["labels"]),
        confidence_json=json.dumps(out["labels"]),
        heatmap_path=heatmap_rel,
        mask_path=mask_rel,
        report_text=None,
        severity_score=float(out["severity_score"]),
    )
    db.add(pred)
    await db.commit()
    await db.refresh(pred)

    base = settings.api_prefix.rstrip("/")
    heatmap_url = f"{base}/heatmap/{pred.id}" if heatmap_rel else None
    mask_url = f"{base}/segmentation/{pred.id}/image" if mask_rel else None

    return PredictResponse(
        study_id=study.id,
        modality=body.modality,
        labels=out["labels"],
        top_findings=out["top_findings"],
        heatmap_url=heatmap_url,
        mask_url=mask_url,
        severity_score=float(out["severity_score"]),
        prediction_id=pred.id,
    )


@router.get("/heatmap/{prediction_id}")
async def get_heatmap(
    prediction_id: int,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> FileResponse:
    pred = await _get_prediction_for_user(db, user.id, prediction_id)
    if not pred.heatmap_path:
        raise HTTPException(status_code=404, detail="No heatmap for this prediction")
    fp = settings.heatmap_dir / pred.heatmap_path
    if not fp.exists():
        raise HTTPException(status_code=404, detail="Heatmap file missing")
    return FileResponse(fp, media_type="image/png")


@router.post("/segmentation", response_model=SegmentationResponse)
async def segmentation_only(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    study_id: int = Form(...),
    modality: str = Form(..., pattern="^(chest_xray|brain_mri)$"),
) -> SegmentationResponse:
    """Re-run or generate segmentation mask overlay for a study."""
    _ensure_dirs()
    study = await _get_study(db, user.id, study_id)
    path = Path(study.stored_path)
    pil = load_image_rgb(path)
    mk_name = f"mask_only_{study.id}.png"
    mk_path = settings.mask_dir / mk_name
    ml_service.segment(pil, modality, mk_path)

    pred = PredictionRecord(
        study_id=study.id,
        labels_json="{}",
        confidence_json="{}",
        heatmap_path=None,
        mask_path=mk_name,
        report_text=None,
        severity_score=0.0,
    )
    db.add(pred)
    await db.commit()
    await db.refresh(pred)

    base = settings.api_prefix.rstrip("/")
    return SegmentationResponse(study_id=study.id, mask_url=f"{base}/segmentation/{pred.id}/image")


@router.get("/segmentation/{prediction_id}/image")
async def get_segmentation_image(
    prediction_id: int,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> FileResponse:
    pred = await _get_prediction_for_user(db, user.id, prediction_id)
    if not pred.mask_path:
        raise HTTPException(status_code=404, detail="No mask for this prediction")
    fp = settings.mask_dir / pred.mask_path
    if not fp.exists():
        raise HTTPException(status_code=404, detail="Mask file missing")
    return FileResponse(fp, media_type="image/png")


@router.post("/generate-report", response_model=ReportResponse)
async def generate_report(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    body: ReportRequest,
) -> ReportResponse:
    pred = await _get_prediction_for_user(db, user.id, body.prediction_id, load_study=True)
    labels = json.loads(pred.labels_json)
    modality = pred.study.modality  # type: ignore[union-attr]
    top = sorted(labels.items(), key=lambda x: x[1], reverse=True)[:3]
    top_findings = [f"{k} ({v:.2f})" for k, v in top]
    structured = report_service.build_structured_report(
        modality=modality,
        labels=labels,
        severity_score=float(pred.severity_score),
        top_findings=top_findings,
    )
    llm_text = None
    if body.use_llm:
        llm_text = await report_service.maybe_llm_explanation(structured)

    text_blob = json.dumps({**structured, "llm": llm_text})
    pred.report_text = text_blob
    await db.commit()

    return ReportResponse(
        prediction_id=pred.id,
        findings=structured["findings"],
        impression=structured["impression"],
        severity_score=float(structured["severity_score"]),
        suggested_action=structured["suggested_action"],
        llm_explanation=llm_text,
    )


@router.get("/stats")
async def stats(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict[str, Any]:
    """Aggregate counts for dashboard charts (per authenticated user)."""
    q = (
        select(Study.modality, func.count(PredictionRecord.id))
        .join(PredictionRecord, PredictionRecord.study_id == Study.id)
        .where(Study.user_id == user.id)
        .group_by(Study.modality)
    )
    rows = (await db.execute(q)).all()
    by_mod = {m: int(c) for m, c in rows}
    q2 = select(func.avg(PredictionRecord.severity_score)).join(Study).where(Study.user_id == user.id)
    avg_sev = (await db.execute(q2)).scalar()
    return {
        "predictions_by_modality": by_mod,
        "average_severity": float(avg_sev or 0.0),
        "total_predictions": sum(by_mod.values()),
    }


@router.get("/history", response_model=list[HistoryItem])
async def history(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = 50,
) -> list[HistoryItem]:
    q = (
        select(PredictionRecord)
        .join(Study)
        .where(Study.user_id == user.id)
        .options(selectinload(PredictionRecord.study))
        .order_by(PredictionRecord.created_at.desc())
        .limit(min(limit, 200))
    )
    rows = (await db.execute(q)).scalars().all()
    items: list[HistoryItem] = []
    for p in rows:
        labels = json.loads(p.labels_json) if p.labels_json else {}
        items.append(
            HistoryItem(
                prediction_id=p.id,
                study_id=p.study_id,
                modality=p.study.modality,  # type: ignore[union-attr]
                created_at=p.created_at.isoformat() if p.created_at else "",
                labels=labels,
                severity_score=float(p.severity_score),
            )
        )
    return items


async def _get_study(db: AsyncSession, user_id: int, study_id: int) -> Study:
    r = await db.execute(select(Study).where(Study.id == study_id, Study.user_id == user_id))
    s = r.scalar_one_or_none()
    if not s:
        raise HTTPException(status_code=404, detail="Study not found")
    return s


async def _get_prediction_for_user(
    db: AsyncSession, user_id: int, prediction_id: int, load_study: bool = False
) -> PredictionRecord:
    q = (
        select(PredictionRecord)
        .join(Study)
        .where(PredictionRecord.id == prediction_id, Study.user_id == user_id)
    )
    if load_study:
        q = q.options(selectinload(PredictionRecord.study))
    pred = (await db.execute(q)).scalar_one_or_none()
    if not pred:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return pred
