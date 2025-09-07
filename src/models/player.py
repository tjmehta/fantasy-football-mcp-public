"""
Player models for fantasy football analysis.

This module contains Pydantic models for representing NFL players,
their statistics, projections, and fantasy-relevant information.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class Position(str, Enum):
    """NFL player positions for fantasy football."""
    QB = "QB"
    RB = "RB"
    WR = "WR"
    TE = "TE"
    K = "K"
    DEF = "DEF"


class InjuryStatus(str, Enum):
    """Player injury status designations."""
    HEALTHY = "Healthy"
    QUESTIONABLE = "Questionable"
    DOUBTFUL = "Doubtful"
    OUT = "Out"
    IR = "IR"
    PUP = "PUP"
    COVID = "COVID-19"
    SUSPENDED = "Suspended"


class Team(str, Enum):
    """NFL team abbreviations."""
    ARI = "ARI"
    ATL = "ATL"
    BAL = "BAL"
    BUF = "BUF"
    CAR = "CAR"
    CHI = "CHI"
    CIN = "CIN"
    CLE = "CLE"
    DAL = "DAL"
    DEN = "DEN"
    DET = "DET"
    GB = "GB"
    HOU = "HOU"
    IND = "IND"
    JAX = "JAX"
    KC = "KC"
    LV = "LV"
    LAC = "LAC"
    LAR = "LAR"
    MIA = "MIA"
    MIN = "MIN"
    NE = "NE"
    NO = "NO"
    NYG = "NYG"
    NYJ = "NYJ"
    PHI = "PHI"
    PIT = "PIT"
    SF = "SF"
    SEA = "SEA"
    TB = "TB"
    TEN = "TEN"
    WAS = "WAS"

    @classmethod
    def _missing_(cls, value):
        """Handle case-insensitive team abbreviations from Yahoo API."""
        if isinstance(value, str):
            # Handle common Yahoo API variations
            value_upper = value.upper()
            for member in cls:
                if member.value == value_upper:
                    return member
        return None


class PlayerStats(BaseModel):
    """Statistical data for a player."""

    # Passing stats
    passing_attempts: Optional[int] = Field(None, ge=0, description="Pass attempts")
    passing_completions: Optional[int] = Field(None, ge=0, description="Pass completions")
    passing_yards: Optional[int] = Field(None, description="Passing yards")
    passing_touchdowns: Optional[int] = Field(None, ge=0, description="Passing touchdowns")
    interceptions: Optional[int] = Field(None, ge=0, description="Interceptions thrown")

    # Rushing stats
    rushing_attempts: Optional[int] = Field(None, ge=0, description="Rush attempts")
    rushing_yards: Optional[int] = Field(None, description="Rushing yards")
    rushing_touchdowns: Optional[int] = Field(None, ge=0, description="Rushing touchdowns")

    # Receiving stats
    targets: Optional[int] = Field(None, ge=0, description="Pass targets")
    receptions: Optional[int] = Field(None, ge=0, description="Receptions")
    receiving_yards: Optional[int] = Field(None, description="Receiving yards")
    receiving_touchdowns: Optional[int] = Field(None, ge=0, description="Receiving touchdowns")

    # Kicking stats
    field_goals_made: Optional[int] = Field(None, ge=0, description="Field goals made")
    field_goals_attempted: Optional[int] = Field(None, ge=0, description="Field goals attempted")
    extra_points_made: Optional[int] = Field(None, ge=0, description="Extra points made")
    extra_points_attempted: Optional[int] = Field(None, ge=0, description="Extra points attempted")

    # Defense stats
    sacks: Optional[Decimal] = Field(None, ge=0, description="Sacks")
    interceptions_def: Optional[int] = Field(None, ge=0, description="Defensive interceptions")
    fumble_recoveries: Optional[int] = Field(None, ge=0, description="Fumble recoveries")
    touchdowns_def: Optional[int] = Field(None, ge=0, description="Defensive touchdowns")
    points_allowed: Optional[int] = Field(None, ge=0, description="Points allowed by defense")
    yards_allowed: Optional[int] = Field(None, ge=0, description="Yards allowed by defense")

    # General stats
    fumbles_lost: Optional[int] = Field(None, ge=0, description="Fumbles lost")
    fantasy_points: Optional[Decimal] = Field(None, description="Fantasy points scored")

    # Game context
    games_played: Optional[int] = Field(None, ge=0, description="Games played")
    games_started: Optional[int] = Field(None, ge=0, description="Games started")

    @validator('passing_completions')
    def completions_not_greater_than_attempts(cls, v, values):
        """Validate that completions don't exceed attempts."""
        if v is not None and 'passing_attempts' in values:
            attempts = values['passing_attempts']
            if attempts is not None and v > attempts:
                raise ValueError('Completions cannot exceed attempts')
        return v

    @validator('receptions')
    def receptions_not_greater_than_targets(cls, v, values):
        """Validate that receptions don't exceed targets."""
        if v is not None and 'targets' in values:
            targets = values['targets']
            if targets is not None and v > targets:
                raise ValueError('Receptions cannot exceed targets')
        return v


class PlayerProjections(BaseModel):
    """Projected statistics and fantasy points for a player."""

    # Core projections
    projected_fantasy_points: Decimal = Field(..., description="Projected fantasy points")
    projected_stats: PlayerStats = Field(..., description="Projected statistical performance")

    # Projection metadata
    confidence_score: Decimal = Field(
        ...,
        ge=0,
        le=1,
        description="Confidence in projection (0-1 scale)"
    )
    projection_source: str = Field(..., description="Source of the projection")
    last_updated: datetime = Field(..., description="When projection was last updated")

    # Variance and risk
    ceiling_points: Optional[Decimal] = Field(None, description="Optimistic projection")
    floor_points: Optional[Decimal] = Field(None, description="Conservative projection")
    bust_probability: Optional[Decimal] = Field(
        None,
        ge=0,
        le=1,
        description="Probability of significant underperformance"
    )

    # Matchup context
    matchup_rating: Optional[str] = Field(None, description="Opponent matchup rating")
    weather_impact: Optional[str] = Field(None, description="Weather impact assessment")

    @validator('ceiling_points')
    def ceiling_greater_than_projection(cls, v, values):
        """Validate ceiling is greater than base projection."""
        if v is not None and 'projected_fantasy_points' in values:
            projection = values['projected_fantasy_points']
            if v < projection:
                raise ValueError('Ceiling must be greater than base projection')
        return v

    @validator('floor_points')
    def floor_less_than_projection(cls, v, values):
        """Validate floor is less than base projection."""
        if v is not None and 'projected_fantasy_points' in values:
            projection = values['projected_fantasy_points']
            if v > projection:
                raise ValueError('Floor must be less than base projection')
        return v


class InjuryReport(BaseModel):
    """Detailed injury information for a player."""

    status: InjuryStatus = Field(..., description="Current injury status")
    injury_type: Optional[str] = Field(None, description="Type of injury")
    body_part: Optional[str] = Field(None, description="Injured body part")
    estimated_return: Optional[datetime] = Field(None, description="Estimated return date")
    practice_participation: Optional[str] = Field(None, description="Practice participation level")
    last_updated: datetime = Field(..., description="When injury report was updated")
    severity_score: Optional[int] = Field(
        None,
        ge=1,
        le=10,
        description="Injury severity (1=minor, 10=season-ending)"
    )
    impact_on_performance: Optional[str] = Field(None, description="Expected impact on performance")


class PlayerValue(BaseModel):
    """Market value and ownership metrics for a player."""

    # Salary information
    draftkings_salary: Optional[int] = Field(None, ge=0, description="DraftKings salary")
    fanduel_salary: Optional[int] = Field(None, ge=0, description="FanDuel salary")
    yahoo_salary: Optional[int] = Field(None, ge=0, description="Yahoo salary")

    # Ownership percentages
    ownership_percentage: Optional[Decimal] = Field(
        None,
        ge=0,
        le=100,
        description="Public ownership percentage"
    )
    projected_ownership: Optional[Decimal] = Field(
        None,
        ge=0,
        le=100,
        description="Projected ownership percentage"
    )

    # Value metrics
    points_per_dollar: Optional[Decimal] = Field(None, description="Points per salary dollar")
    value_score: Optional[Decimal] = Field(
        None,
        ge=0,
        le=10,
        description="Overall value score (0-10)"
    )

    # Market trends
    salary_change: Optional[int] = Field(None, description="Salary change from previous week")
    ownership_trend: Optional[str] = Field(None, description="Ownership trend direction")

    last_updated: datetime = Field(..., description="When values were last updated")


class Player(BaseModel):
    """Comprehensive player model for fantasy football analysis."""

    # Basic information
    id: str = Field(..., description="Unique player identifier")
    name: str = Field(..., min_length=1, description="Player's full name")
    position: Position = Field(..., description="Player's position")
    team: Team = Field(..., description="Player's NFL team")

    # Physical attributes
    age: Optional[int] = Field(None, ge=18, le=50, description="Player's age")
    height: Optional[str] = Field(None, description="Player's height")
    weight: Optional[int] = Field(None, ge=150, le=400, description="Player's weight in pounds")

    # Experience
    years_pro: Optional[int] = Field(None, ge=0, description="Years in the NFL")
    college: Optional[str] = Field(None, description="College attended")

    # Season information
    season: int = Field(..., ge=2020, description="NFL season year")
    week: Optional[int] = Field(None, ge=1, le=18, description="Current NFL week")

    # Performance data
    season_stats: Optional[PlayerStats] = Field(None, description="Season statistics")
    last_game_stats: Optional[PlayerStats] = Field(None, description="Previous game statistics")
    career_stats: Optional[PlayerStats] = Field(None, description="Career statistics")

    # Projections and analysis
    projections: Optional[PlayerProjections] = Field(None, description="Performance projections")

    # Health and availability
    injury_report: Optional[InjuryReport] = Field(None, description="Injury information")
    is_active: bool = Field(True, description="Whether player is on active roster")

    # Value and market data
    value_metrics: Optional[PlayerValue] = Field(None, description="Value and ownership metrics")

    # Matchup context
    opponent: Optional[Team] = Field(None, description="Next opponent")
    home_away: Optional[str] = Field(None, description="Home/Away designation")

    # Additional context
    news_notes: Optional[List[str]] = Field(None, description="Recent news and notes")
    tags: Optional[List[str]] = Field(None, description="Player tags (e.g., 'sleeper', 'must-start')")

    # Metadata
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    data_source: Optional[str] = Field(None, description="Source of player data")

    class Config:
        """Pydantic model configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: float(v)
        }
        use_enum_values = True
        validate_assignment = True

    def get_display_name(self) -> str:
        """Get formatted display name for UI."""
        return f"{self.name} ({self.position.value} - {self.team.value})"

    def is_injured(self) -> bool:
        """Check if player has injury concerns."""
        if not self.injury_report:
            return False
        return self.injury_report.status not in [InjuryStatus.HEALTHY]

    def get_fantasy_points_per_game(self) -> Optional[Decimal]:
        """Calculate fantasy points per game for the season."""
        if not self.season_stats or not self.season_stats.games_played:
            return None
        if self.season_stats.games_played == 0:
            return None
        if not self.season_stats.fantasy_points:
            return None
        return self.season_stats.fantasy_points / self.season_stats.games_played

    def get_projected_value(self) -> Optional[Decimal]:
        """Get projected points per dollar value."""
        if not self.projections or not self.value_metrics:
            return None

        # Use DraftKings salary as default
        salary = (self.value_metrics.draftkings_salary or
                 self.value_metrics.fanduel_salary or
                 self.value_metrics.yahoo_salary)

        if not salary or salary == 0:
            return None

        return self.projections.projected_fantasy_points / (salary / 1000)


class PlayerSearchFilter(BaseModel):
    """Filters for searching and filtering players."""

    positions: Optional[List[Position]] = Field(None, description="Filter by positions")
    teams: Optional[List[Team]] = Field(None, description="Filter by teams")
    min_projected_points: Optional[Decimal] = Field(None, description="Minimum projected points")
    max_salary: Optional[int] = Field(None, description="Maximum salary")
    injury_status: Optional[List[InjuryStatus]] = Field(None, description="Filter by injury status")
    max_ownership: Optional[Decimal] = Field(None, description="Maximum ownership percentage")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")

    class Config:
        """Pydantic model configuration."""
        use_enum_values = True


class PlayerComparison(BaseModel):
    """Model for comparing multiple players."""

    players: List[Player] = Field(..., min_items=2, description="Players to compare")
    comparison_metrics: List[str] = Field(..., description="Metrics to compare")
    winner: Optional[str] = Field(None, description="Recommended player ID")
    reasoning: Optional[str] = Field(None, description="Comparison reasoning")

    created_at: datetime = Field(default_factory=datetime.utcnow, description="Comparison timestamp")

    class Config:
        """Pydantic model configuration."""
        json_encoders = {datetime: lambda v: v.isoformat()}
