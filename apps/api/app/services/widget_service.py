import secrets

from sqlalchemy.orm import Session

from app.models.user import User
from app.models.widget import WidgetConfig
from app.repositories.audit_repository import AuditRepository
from app.repositories.widget_repository import WidgetRepository
from app.schemas.widget import WidgetCreate, WidgetUpdate

#Business logic for creating, updating, listing, and publicly loading widget configs. It also writes audit logs for widget changes.
class WidgetError(RuntimeError):
    pass


class WidgetService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.widgets = WidgetRepository(db)
        self.audit = AuditRepository(db)

    def list_widgets(self) -> list[WidgetConfig]:
        return self.widgets.list_widgets()

    def get_public_config(self, widget_id: str) -> WidgetConfig:
        widget = self.widgets.get_by_widget_id(widget_id)
        if widget is None or not widget.is_active:
            raise WidgetError("Widget not found or disabled.")
        return widget

    def create_widget(self, *, payload: WidgetCreate, actor: User) -> WidgetConfig:
        widget_id = payload.widget_id or secrets.token_urlsafe(12)
        if self.widgets.get_by_widget_id(widget_id) is not None:
            raise WidgetError("A widget with this widget_id already exists.")

        widget = WidgetConfig(
            widget_id=widget_id,
            name=payload.name,
            allowed_origins=payload.allowed_origins,
            theme=payload.theme,
            greeting=payload.greeting,
            enabled_tools=payload.enabled_tools,
            is_active=payload.is_active,
            created_by_user_id=actor.id,
        )
        self.widgets.create(widget)
        self.audit.create(
            actor=actor.email,
            action="widget.create",
            target=f"widget:{widget.widget_id}",
        )
        self.db.commit()
        self.db.refresh(widget)
        return widget

    def update_widget(
        self,
        *,
        widget_id: str,
        payload: WidgetUpdate,
        actor: User,
    ) -> WidgetConfig:
        widget = self.widgets.get_by_widget_id(widget_id)
        if widget is None:
            raise WidgetError("Widget not found.")

        update_data = payload.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(widget, key, value)

        self.audit.create(
            actor=actor.email,
            action="widget.update",
            target=f"widget:{widget.widget_id}",
        )
        self.db.commit()
        self.db.refresh(widget)
        return widget