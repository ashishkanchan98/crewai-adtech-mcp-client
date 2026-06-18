from datetime import datetime

from sqlalchemy import DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class CrewTicket(Base):
    __tablename__ = "crew_tickets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    advertiser_id: Mapped[str] = mapped_column(String(50), nullable=True)
    campaign_id: Mapped[str] = mapped_column(String(50), nullable=True)
    use_case: Mapped[str] = mapped_column(String(60), nullable=True)    # which crew was triggered
    agents_used: Mapped[str] = mapped_column(Text, nullable=True)       # comma-separated agent roles
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False)     # RESOLVED_BY_CREW | ESCALATED
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
