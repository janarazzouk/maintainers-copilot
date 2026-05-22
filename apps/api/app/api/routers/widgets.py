from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import require_admin
from app.infra.database import get_db
from app.models.user import User
from app.schemas.widget import PublicWidgetConfig, WidgetCreate, WidgetResponse, WidgetUpdate
from app.services.widget_service import WidgetError, WidgetService

#Admin-only:

#GET   /admin/widgets
#POST  /admin/widgets
#PATCH /admin/widgets/{widget_id}

#Public:

#GET /widgets/{widget_id}/config


admin_router = APIRouter(prefix="/admin/widgets", tags=["admin-widgets"])
public_router = APIRouter(prefix="/widgets", tags=["widgets"])


@admin_router.get("", response_model=list[WidgetResponse])
def list_widgets(
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> list[WidgetResponse]:
    widgets = WidgetService(db).list_widgets()
    return [WidgetResponse.model_validate(widget) for widget in widgets]


@admin_router.post("", response_model=WidgetResponse, status_code=status.HTTP_201_CREATED)
def create_widget(
    payload: WidgetCreate,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
) -> WidgetResponse:
    try:
        widget = WidgetService(db).create_widget(payload=payload, actor=admin)
    except WidgetError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "WIDGET_CREATE_FAILED", "message": str(exc)},
        ) from exc

    return WidgetResponse.model_validate(widget)


@admin_router.patch("/{widget_id}", response_model=WidgetResponse)
def update_widget(
    widget_id: str,
    payload: WidgetUpdate,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
) -> WidgetResponse:
    try:
        widget = WidgetService(db).update_widget(
            widget_id=widget_id,
            payload=payload,
            actor=admin,
        )
    except WidgetError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "WIDGET_NOT_FOUND", "message": str(exc)},
        ) from exc

    return WidgetResponse.model_validate(widget)


@public_router.get("/{widget_id}/config", response_model=PublicWidgetConfig)
def get_widget_config(
    widget_id: str,
    db: Session = Depends(get_db),
) -> PublicWidgetConfig:
    try:
        widget = WidgetService(db).get_public_config(widget_id)
    except WidgetError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "WIDGET_NOT_FOUND", "message": str(exc)},
        ) from exc

    return PublicWidgetConfig(
        widget_id=widget.widget_id,
        theme=widget.theme,
        greeting=widget.greeting,
        enabled_tools=widget.enabled_tools,
    )