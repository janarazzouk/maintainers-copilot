from sqlalchemy.orm import Session

from app.models.audit_log import AuditLog

#Writes audit log rows. This is used when users register and when widget configs are created/updated.
class AuditRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, *, actor: str, action: str, target: str) -> AuditLog:
        row = AuditLog(actor=actor, action=action, target=target)
        self.db.add(row)
        self.db.flush()
        return row