from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.user import User


class UserRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def count_users(self) -> int:
        return self.db.scalar(select(func.count()).select_from(User)) or 0

    def get_by_email(self, email: str) -> User | None:
        return self.db.scalar(select(User).where(User.email == email.lower()))

    def get_by_id(self, user_id: int) -> User | None:
        return self.db.get(User, user_id)

    def create_user(
        self,
        *,
        email: str,
        hashed_password: str,
        role: str = "user",
    ) -> User:
        user = User(
            email=email.lower(),
            hashed_password=hashed_password,
            role=role,
        )
        self.db.add(user)
        self.db.flush()
        return user