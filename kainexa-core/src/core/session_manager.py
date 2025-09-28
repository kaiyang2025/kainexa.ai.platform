# src/core/session_manager.py 생성

"""
세션 관리자
"""
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
import jwt
import re

from src.core.models import User, Session, Conversation, Message
from src.core.config import settings

class SessionManager:
    """세션 관리"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.secret_key = settings.SECRET_KEY
        self.token_expiry = settings.ACCESS_TOKEN_EXPIRE_MINUTES
        
    async def create_session(self, 
                            user_id: str,
                            ip_address: str = None,
                            user_agent: str = None) -> Dict[str, Any]:
        """새 세션 생성"""
        
        # 토큰 생성
        session_token = secrets.token_urlsafe(32) # 256-bit 난수
        now = datetime.now(timezone.utc)          # UTC 고정 (timezone-aware)
        expires_at = now + timedelta(minutes=self.token_expiry)
        
        # DB에 세션 저장
        session = Session(
            user_id=user_id,
            session_token=session_token,
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=expires_at
        )
        
        self.db.add(session)
        await self.db.commit()
        await self.db.refresh(session)
        
        # JWT 토큰 생성
        jwt_token = self._create_jwt_token(session)
        
        return {
            "session_id": str(session.id),
            "session_token": session_token,
            "jwt_token": jwt_token,
            "expires_at": expires_at.isoformat()
        }
    
    async def get_session(self, session_token: str) -> Optional[Session]:
        """세션 조회"""
        
        query = select(Session).where(
            and_(
                Session.session_token == session_token,
                Session.expires_at > datetime.now(timezone.utc)
            )
        )
        
        result = await self.db.execute(query)
        session = result.scalar_one_or_none()
        
        if session:
            # 마지막 활동 시간 업데이트
            session.last_activity = datetime.now(timezone.utc)
            await self.db.commit()
        
        return session
    
    async def validate_session(self, session_token: str) -> bool:
        """세션 유효성 검증"""
        session = await self.get_session(session_token)
        return session is not None
    
    async def extend_session(self, session_id: str, minutes: int = 30):
        """세션 연장"""
        
        query = select(Session).where(Session.id == session_id)
        result = await self.db.execute(query)
        session = result.scalar_one_or_none()
        
        if session:
            session.expires_at = datetime.now(timezone.utc) + timedelta(minutes=minutes)
            session.last_activity = datetime.now(timezone.utc)
            await self.db.commit()
        
        return session
    
    async def end_session(self, session_id: str):
        """세션 종료"""
        
        query = select(Session).where(Session.id == session_id)
        result = await self.db.execute(query)
        session = result.scalar_one_or_none()
        
        if session:
            await self.db.delete(session)
            await self.db.commit()
    
    async def cleanup_expired_sessions(self):
        """만료된 세션 정리"""
        
        query = select(Session).where(Session.expires_at <= datetime.now(timezone.utc))
        result = await self.db.execute(query)
        expired_sessions = result.scalars().all()
        
        for session in expired_sessions:
            await self.db.delete(session)
        
        await self.db.commit()
        
        return len(expired_sessions)
    
    def _create_jwt_token(self, session: Session) -> str:
        """JWT 토큰 생성"""
        
        payload = {
            "session_id": str(session.id),
            "user_id": str(session.user_id),
            "exp": session.expires_at,
            "iat": datetime.now(timezone.utc)
        }
        
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
    
    def verify_jwt_token(self, token: str) -> Optional[Dict]:
        """JWT 토큰 검증"""
        
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    async def get_or_create_user(self, email: str):
        from sqlalchemy import select
        from src.core.models import User
        res = await self.db.execute(select(User).where(User.email == email))
        user = res.scalar_one_or_none()
        if user is None:
            user = await self.create_user(email=email)
        return user

    async def get_or_create_session(self, user_id):
        from sqlalchemy import select
        from src.core.models import Session
        res = await self.db.execute(select(Session).where(Session.user_id == user_id))
        session = res.scalar_one_or_none()
        if session is None:
            session = await self.create_session(user_id=user_id)
        return session

    async def ensure_user_and_session(self, user_email: str):
        user = await self.get_or_create_user(user_email)
        session = await self.get_or_create_session(user.id)
        return user, session
    
    async def create_user(self, email: str, role: str = "user") -> User:
        """
        - username NOT NULL 대응: email에서 username 생성
        - 모델에 존재하는 컬럼만 안전하게 세팅
        """
        # 1) username 생성 (email의 @앞부분 → 소문자/영문숫자._- 만 허용, 100자 제한)
        base = email.split("@")[0] if "@" in email else email
        candidate = re.sub(r"[^a-zA-Z0-9._-]", "_", base).lower()[:100] or "user"

        kwargs = {"email": email}
        if hasattr(User, "username"):
            kwargs["username"] = candidate
        if hasattr(User, "full_name"):
            kwargs["full_name"] = base  # 풀네임은 간단히 base로
        if hasattr(User, "role"):
            kwargs["role"] = role
        if hasattr(User, "is_active"):
            kwargs["is_active"] = True

        user = User(**kwargs)
       
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        return user
    

class ConversationManager:
    """대화 관리"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_conversation(self,
                                 session_id: str,
                                 user_id: str,
                                 title: str = None) -> Conversation:
        """새 대화 생성"""
        
        conversation = Conversation(
            session_id=session_id,
            user_id=user_id,
            title=title or f"대화 {datetime.now(timezone.utc):%Y-%m-%d %H:%M}"
        )
        
        self.db.add(conversation)
        await self.db.commit()
        await self.db.refresh(conversation)
        
        return conversation
    
    async def add_message(self,
                         conversation_id: str,
                         role: str,
                         content: str,
                         meta: Dict = None) -> Message:
        """메시지 추가"""
        
        # 토큰 수 계산 (간단한 추정)
        tokens = len(content.split()) * 1.3
        
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            meta=meta or {},
            tokens=int(tokens)
        )
        
        self.db.add(message)
        
        # 대화 업데이트
        query = select(Conversation).where(Conversation.id == conversation_id)
        result = await self.db.execute(query)
        conversation = result.scalar_one_or_none()
        
        if conversation:
            conversation.updated_at = datetime.now(timezone.utc)
        
        await self.db.commit()
        await self.db.refresh(message)
        
        return message
    
    async def get_conversation_history(self,
                                      conversation_id: str,
                                      limit: int = 10) -> List[Message]:
        """대화 기록 조회"""
        
        query = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        
        result = await self.db.execute(query)
        messages = result.scalars().all()
        
        return list(reversed(messages))
    
    async def get_user_conversations(self,
                                    user_id: str,
                                    limit: int = 10) -> List[Conversation]:
        """사용자 대화 목록 조회"""
        
        query = (
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
        )
        
        result = await self.db.execute(query)
        conversations = result.scalars().all()
        
        return conversations
    
    async def update_conversation_context(self,
                                         conversation_id: str,
                                         context: Dict):
        """대화 컨텍스트 업데이트"""
        
        query = select(Conversation).where(Conversation.id == conversation_id)
        result = await self.db.execute(query)
        conversation = result.scalar_one_or_none()
        
        if conversation:
            conversation.context = context
            conversation.updated_at = datetime.now()
            await self.db.commit()
        
        return conversation