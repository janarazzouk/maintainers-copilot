from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.widget import WidgetConfig


class WidgetRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def list_widgets(self) -> list[WidgetConfig]:
        statement = select(WidgetConfig).order_by(WidgetConfig.created_at.desc())
        return list(self.db.scalars(statement).all())

    def get_by_widget_id(self, widget_id: str) -> WidgetConfig | None:
        return self.db.scalar(
            select(WidgetConfig).where(WidgetConfig.widget_id == widget_id)
        )

    def create(self, widget: WidgetConfig) -> WidgetConfig:
        self.db.add(widget)
        self.db.flush()
        return widget