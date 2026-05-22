from sqlalchemy.orm import Session

from app.infra.security import create_access_token, hash_password, verify_password
from app.models.user import User
from app.repositories.audit_repository import AuditRepository
from app.repositories.user_repository import UserRepository

#Business logic for registering users, logging in, issuing JWTs, and deciding the first-user-is-admin rule.
class AuthError(RuntimeError):
    pass


class AuthService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.users = UserRepository(db)
        self.audit = AuditRepository(db)

    def register_user(self, *, email: str, password: str) -> User:
        normalized_email = email.strip().lower()
        if not normalized_email:
            raise AuthError("Email is required.")
        if len(password) < 8:
            raise AuthError("Password must be at least 8 characters.")
        if self.users.get_by_email(normalized_email) is not None:
            raise AuthError("A user with this email already exists.")

        # Bootstrap rule: the first registered account becomes admin.
        role = "admin" if self.users.count_users() == 0 else "user"

        user = self.users.create_user(
            email=normalized_email,
            hashed_password=hash_password(password),
            role=role,
        )
        self.audit.create(
            actor=normalized_email,
            action="user.register",
            target=f"user:{user.id}:{role}",
        )
        self.db.commit()
        self.db.refresh(user)
        return user

    def authenticate(self, *, email: str, password: str) -> User:
        user = self.users.get_by_email(email.strip().lower())
        if user is None or not verify_password(password, user.hashed_password):
            raise AuthError("Invalid email or password.")
        if not user.is_active:
            raise AuthError("This user is inactive.")
        return user

    def issue_access_token(
        self,
        *,
        user: User,
        secret_key: str,
        expires_minutes: int,
    ) -> str:
        return create_access_token(
            subject=str(user.id),
            role=user.role,
            secret_key=secret_key,
            expires_minutes=expires_minutes,
        )