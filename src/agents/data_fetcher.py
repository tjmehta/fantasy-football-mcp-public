"""
Yahoo Fantasy Sports API Data Fetcher Agent.

This module provides the DataFetcherAgent class that handles all Yahoo Fantasy Sports
API interactions including OAuth2 authentication, data fetching with rate limiting,
and intelligent caching through the cache manager.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
import time

import aiohttp
from loguru import logger
from yfpy import YahooFantasySportsQuery
from yfpy.models import Game, League, Team, Player as YfpyPlayer, Roster, Matchup

from config.settings import Settings
from ..models.player import Player, Position, Team as NFLTeam, InjuryReport, InjuryStatus, PlayerStats
from ..models.matchup import Matchup as FantasyMatchup, GameStatus
from ..models.lineup import Lineup
from .cache_manager import CacheManagerAgent


class APIEndpoint(str, Enum):
    """Yahoo Fantasy Sports API endpoints."""
    USER_LEAGUES = "user_leagues"
    LEAGUE_INFO = "league_info"
    TEAM_ROSTER = "team_roster"
    TEAM_MATCHUP = "team_matchup"
    PLAYER_INFO = "player_info"
    AVAILABLE_PLAYERS = "available_players"
    INJURY_REPORT = "injury_report"
    LEAGUE_STANDINGS = "league_standings"
    LEAGUE_TRANSACTIONS = "league_transactions"


class RateLimitError(Exception):
    """Exception raised when API rate limit is exceeded."""
    pass


class AuthenticationError(Exception):
    """Exception raised when authentication fails."""
    pass


@dataclass
class APIRequest:
    """API request wrapper with retry logic."""
    endpoint: APIEndpoint
    params: Dict[str, Any]
    attempt: int = 0
    max_retries: int = 3
    backoff_factor: float = 2.0
    timeout: int = 30


@dataclass
class RateLimitTracker:
    """Track API rate limiting."""
    requests_per_window: int = 100
    window_seconds: int = 3600
    requests_made: int = 0
    window_start: datetime = None

    def __post_init__(self):
        if self.window_start is None:
            self.window_start = datetime.utcnow()

    def can_make_request(self) -> bool:
        """Check if we can make another request within rate limits."""
        now = datetime.utcnow()

        # Reset window if expired
        if now - self.window_start > timedelta(seconds=self.window_seconds):
            self.requests_made = 0
            self.window_start = now

        return self.requests_made < self.requests_per_window

    def record_request(self) -> None:
        """Record a successful API request."""
        self.requests_made += 1

    def time_until_reset(self) -> timedelta:
        """Get time until rate limit window resets."""
        window_end = self.window_start + timedelta(seconds=self.window_seconds)
        remaining = window_end - datetime.utcnow()
        return remaining if remaining.total_seconds() > 0 else timedelta(0)


class DataFetcherAgent:
    """
    Agent responsible for fetching data from Yahoo Fantasy Sports API.

    This agent handles:
    - OAuth2 authentication with Yahoo
    - Rate-limited API requests with retry logic
    - Parallel data fetching for multiple leagues
    - Intelligent caching of API responses
    - Data transformation to internal models
    - Graceful error handling and recovery
    """

    def __init__(self, settings: Settings, cache_manager: CacheManagerAgent):
        """
        Initialize the data fetcher agent.

        Args:
            settings: Application settings containing API configuration
            cache_manager: Cache manager for intelligent caching
        """
        self.settings = settings
        self.cache_manager = cache_manager

        # Rate limiting
        self.rate_limiter = RateLimitTracker(
            requests_per_window=settings.yahoo_api_rate_limit,
            window_seconds=settings.yahoo_api_rate_window_seconds
        )

        # Yahoo API client (initialized on first use)
        self._yahoo_client: Optional[YahooFantasySportsQuery] = None
        self._auth_token: Optional[str] = None
        self._auth_expires: Optional[datetime] = None

        # Session for HTTP requests
        self._session: Optional[aiohttp.ClientSession] = None

        # Semaphore for controlling concurrent requests
        self._semaphore = asyncio.Semaphore(settings.max_workers)

        logger.info("DataFetcherAgent initialized")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()

    async def initialize(self) -> None:
        """Initialize the data fetcher."""
        try:
            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=self.settings.async_timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)

            # Initialize Yahoo API client
            await self._initialize_yahoo_client()

            logger.info("DataFetcherAgent initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize DataFetcherAgent: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if self._session:
                await self._session.close()

            logger.info("DataFetcherAgent cleaned up")

        except Exception as e:
            logger.error(f"Error during DataFetcherAgent cleanup: {e}")

    async def get_user_leagues(self, game_key: str = None) -> List[Dict[str, Any]]:
        """
        Get all leagues for the authenticated user.

        Args:
            game_key: Optional specific game/season (e.g., "nfl.2024")

        Returns:
            List of league information dictionaries
        """
        cache_key = f"user_leagues:{game_key or 'all'}"

        # Check cache first
        cached_data = await self.cache_manager.get(cache_key)
        if cached_data is not None:
            logger.debug(f"Returning cached user leagues for game: {game_key}")
            return cached_data

        try:
            # Make API request
            request = APIRequest(
                endpoint=APIEndpoint.USER_LEAGUES,
                params={"game_key": game_key} if game_key else {}
            )

            leagues_data = await self._make_api_request(request)

            # Transform to our format
            leagues = []
            if leagues_data:
                # Handle both single league and list of leagues
                if not isinstance(leagues_data, list):
                    leagues_data = [leagues_data]

                for league in leagues_data:
                    league_info = {
                        'league_id': getattr(league, 'league_id', None),
                        'league_key': getattr(league, 'league_key', None),
                        'name': getattr(league, 'name', 'Unknown'),
                        'game_key': getattr(league, 'game_key', None),
                        'season': getattr(league, 'season', None),
                        'is_finished': getattr(league, 'is_finished', False),
                        'num_teams': getattr(league, 'num_teams', None),
                        'scoring_type': getattr(league, 'scoring_type', None),
                        'league_type': getattr(league, 'league_type', None),
                        'url': getattr(league, 'url', None)
                    }
                    leagues.append(league_info)

            # Cache the results
            await self.cache_manager.set(
                cache_key,
                leagues,
                ttl=timedelta(hours=4),  # Leagues don't change often
                tags=["user_leagues", "yahoo_api"]
            )

            logger.info(f"Retrieved {len(leagues)} leagues for user")
            return leagues

        except Exception as e:
            logger.error(f"Error getting user leagues: {e}")
            raise

    async def get_user_team_key(self, league_key: str) -> Optional[str]:
        """
        Get the user's team key within a specific league.

        Args:
            league_key: Yahoo league identifier (e.g., "461.l.578668")

        Returns:
            User's team key within the league (e.g., "461.l.578668.t.X") or None if not found
        """
        cache_key = f"user_team_key:{league_key}"

        # Check cache first
        cached_team_key = await self.cache_manager.get(cache_key)
        if cached_team_key is not None:
            logger.debug(f"Returning cached user team key for league {league_key}")
            return cached_team_key

        try:
            # Initialize Yahoo client if needed
            if self._yahoo_client is None:
                await self._initialize_yahoo_client()

            # Set league context
            league_id = league_key.split(".")[-1]
            self._yahoo_client.league_id = league_id

            # Get all teams in the league
            league_teams = self._yahoo_client.get_league_teams()

            if not league_teams:
                logger.warning(f"No teams found in league {league_key}")
                return None

            # For now, we need a way to identify which team belongs to the user
            # This is a simplified approach - in a real implementation, you'd compare
            # manager information or use another method to identify the user's team
            if not isinstance(league_teams, list):
                league_teams = [league_teams]

            # Look for the user's team by checking manager ownership
            # In Yahoo Fantasy, we need to identify which team belongs to the authenticated user
            user_team_key = None

            # First, log all teams for debugging
            logger.info(f"Found {len(league_teams)} teams in league {league_key}:")
            for i, team in enumerate(league_teams):
                team_key = getattr(team, 'team_key', None)
                team_name = getattr(team, 'name', '')
                team_id = team_key.split(".")[-1] if team_key else "unknown"
                logger.info(f"  Team {i+1}: '{team_name}' (Key: {team_key}, ID: {team_id})")

            # Try to get current user GUID and match with team managers
            try:
                logger.info("Getting current user info...")
                current_user = self._yahoo_client.get_current_user()
                user_guid = getattr(current_user, 'guid', None)

                if user_guid:
                    logger.info(f"Current user GUID: {user_guid}")

                    # Check each team's managers for matching GUID
                    for team in league_teams:
                        team_key = getattr(team, 'team_key', None)
                        team_name = getattr(team, 'name', '')

                        # Convert bytes to string if needed
                        team_name_str = team_name.decode('utf-8') if isinstance(team_name, bytes) else str(team_name)

                        # Get team managers
                        managers = getattr(team, 'managers', [])
                        if not isinstance(managers, list):
                            managers = [managers]

                        # Check managers for GUID match
                        for manager in managers:
                            manager_guid = getattr(manager, 'guid', None)
                            if manager_guid == user_guid:
                                user_team_key = team_key
                                logger.info(f"Found user's team by GUID match: {team_name_str} (Key: {team_key})")
                                break

                        if user_team_key:
                            break
                else:
                    logger.warning("Could not get user GUID")

            except Exception as e:
                logger.warning(f"Could not get current user: {e}")

            if user_team_key:
                # Cache the result
                await self.cache_manager.set(cache_key, user_team_key, timedelta(hours=24))
                return user_team_key

            logger.warning(f"Could not determine user's team in league {league_key}")
            return None

        except Exception as e:
            logger.error(f"Error getting user team key for league {league_key}: {e}")
            raise

    async def get_roster(self, league_key: str, team_key: str, week: int = None) -> Dict[str, Any]:
        """
        Get team roster for a specific league and week.

        Args:
            league_key: Yahoo league identifier
            team_key: Yahoo team identifier
            week: Optional week number (current week if not specified)

        Returns:
            Roster information dictionary
        """
        cache_key = f"roster:{league_key}:{team_key}:{week or 'current'}"

        # Check cache first
        cached_data = await self.cache_manager.get(cache_key)
        if cached_data is not None:
            logger.debug(f"Returning cached roster for team {team_key}, week {week}")
            return cached_data

        try:
            # Make API request
            request = APIRequest(
                endpoint=APIEndpoint.TEAM_ROSTER,
                params={
                    "league_key": league_key,
                    "team_key": team_key,
                    "week": week
                }
            )

            roster_data = await self._make_api_request(request)

            # Transform roster data
            roster_info = {
                'team_key': team_key,
                'league_key': league_key,
                'week': week,
                'players': [],
                'roster_positions': {},  # Will be populated from player positions
                'last_updated': datetime.utcnow().isoformat()
            }

            if roster_data and hasattr(roster_data, 'players'):
                position_counts = {}
                for player in roster_data.players:
                    player_info = await self._transform_yahoo_player(player)
                    roster_info['players'].append(player_info)

                    # Track roster positions
                    if hasattr(player, 'selected_position') and player.selected_position:
                        position = getattr(player.selected_position, 'position', 'UNKNOWN')
                        position_counts[position] = position_counts.get(position, 0) + 1

                # Set roster positions based on what we found
                roster_info['roster_positions'] = position_counts

            # Cache with shorter TTL since rosters change frequently
            await self.cache_manager.set(
                cache_key,
                roster_info,
                ttl=timedelta(hours=2),
                tags=["roster", "yahoo_api", f"league:{league_key}"]
            )

            logger.info(f"Retrieved roster for team {team_key}, {len(roster_info['players'])} players")
            return roster_info

        except Exception as e:
            logger.error(f"Error getting roster for team {team_key}: {e}")
            raise

    async def get_matchup(self, league_key: str, team_key: str, week: int) -> Dict[str, Any]:
        """
        Get matchup information for a team in a specific week.

        Args:
            league_key: Yahoo league identifier
            team_key: Yahoo team identifier
            week: Week number

        Returns:
            Matchup information dictionary
        """
        cache_key = f"matchup:{league_key}:{team_key}:{week}"

        # Check cache first
        cached_data = await self.cache_manager.get(cache_key)
        if cached_data is not None:
            logger.debug(f"Returning cached matchup for team {team_key}, week {week}")
            return cached_data

        try:
            # Make API request
            request = APIRequest(
                endpoint=APIEndpoint.TEAM_MATCHUP,
                params={
                    "league_key": league_key,
                    "team_key": team_key,
                    "week": week
                }
            )

            matchup_data = await self._make_api_request(request)

            # Transform matchup data
            matchup_info = {
                'league_key': league_key,
                'week': week,
                'teams': [],
                'is_playoffs': False,
                'is_consolation': False,
                'winner_team_key': None,
                'status': 'upcoming',
                'last_updated': datetime.utcnow().isoformat()
            }

            if matchup_data and hasattr(matchup_data, 'teams'):
                for team in matchup_data.teams:
                    team_info = {
                        'team_key': team.team_key,
                        'name': getattr(team, 'name', ''),
                        'projected_points': getattr(team, 'projected_points', None),
                        'actual_points': getattr(team, 'actual_points', None)
                    }
                    matchup_info['teams'].append(team_info)

            # Set matchup status and winner if available
            if hasattr(matchup_data, 'status'):
                matchup_info['status'] = matchup_data.status
            if hasattr(matchup_data, 'winner_team_key'):
                matchup_info['winner_team_key'] = matchup_data.winner_team_key

            # Cache matchup data
            await self.cache_manager.set(
                cache_key,
                matchup_info,
                ttl=timedelta(hours=1),  # Matchups update during games
                tags=["matchup", "yahoo_api", f"league:{league_key}", f"week:{week}"]
            )

            logger.info(f"Retrieved matchup for team {team_key}, week {week}")
            return matchup_info

        except Exception as e:
            logger.error(f"Error getting matchup for team {team_key}, week {week}: {e}")
            raise

    async def get_player(self, player_key: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information for a specific player.

        Args:
            player_key: Yahoo player identifier

        Returns:
            Player information dictionary or None if not found
        """
        cache_key = f"player:{player_key}"

        # Check cache first
        cached_data = await self.cache_manager.get(cache_key)
        if cached_data is not None:
            logger.debug(f"Returning cached player data for {player_key}")
            return cached_data

        try:
            # Make API request
            request = APIRequest(
                endpoint=APIEndpoint.PLAYER_INFO,
                params={"player_key": player_key}
            )

            player_data = await self._make_api_request(request)

            if not player_data:
                return None

            # Transform player data
            player_info = await self._transform_yahoo_player(player_data)

            # Cache player data with longer TTL (player info doesn't change much)
            await self.cache_manager.set(
                cache_key,
                player_info,
                ttl=timedelta(hours=6),
                tags=["player", "yahoo_api"]
            )

            logger.debug(f"Retrieved player data for {player_key}")
            return player_info

        except Exception as e:
            logger.error(f"Error getting player {player_key}: {e}")
            return None

    async def get_available_players(
        self,
        league_key: str,
        position: str = None,
        status: str = "A",  # A=Available, W=Waivers, T=Taken
        count: int = 25
    ) -> List[Dict[str, Any]]:
        """
        Get available players in a league.

        Args:
            league_key: Yahoo league identifier
            position: Optional position filter (QB, RB, WR, TE, K, DEF)
            status: Player status filter (A=Available, W=Waivers, T=Taken)
            count: Maximum number of players to return

        Returns:
            List of available player information dictionaries
        """
        cache_key = f"available_players:{league_key}:{position or 'all'}:{status}:{count}"

        # Check cache first (shorter TTL since availability changes frequently)
        cached_data = await self.cache_manager.get(cache_key)
        if cached_data is not None:
            logger.debug(f"Returning cached available players for league {league_key}")
            return cached_data

        try:
            # Make API request
            request = APIRequest(
                endpoint=APIEndpoint.AVAILABLE_PLAYERS,
                params={
                    "league_key": league_key,
                    "position": position,
                    "status": status,
                    "count": count
                }
            )

            players_data = await self._make_api_request(request)

            # Debug the raw API response
            logger.debug(f"Raw players_data type: {type(players_data)}")
            logger.debug(f"Raw players_data: {players_data}")
            logger.debug(f"Has 'players' attribute: {hasattr(players_data, 'players') if players_data else False}")
            if players_data and hasattr(players_data, 'players'):
                logger.debug(f"Players count: {len(players_data.players) if players_data.players else 0}")

            # Transform and filter players data
            available_players = []
            if players_data and hasattr(players_data, 'players'):
                logger.debug(f"Processing {len(players_data.players)} players from API")
                for player in players_data.players:
                    player_info = await self._transform_yahoo_player(player)

                    # Apply client-side filtering since API doesn't support it
                    if position and player_info.get('position', '').upper() != position.upper():
                        continue

                    # Note: Status filtering is complex since we'd need ownership data
                    # For now, we'll include all players and let the waiver wire logic handle it

                    available_players.append(player_info)
                logger.debug(f"After filtering: {len(available_players)} players remain")
            else:
                logger.warning("No players_data or no 'players' attribute found")

            # Cache with short TTL since player availability changes rapidly
            await self.cache_manager.set(
                cache_key,
                available_players,
                ttl=timedelta(minutes=30),
                tags=["available_players", "yahoo_api", f"league:{league_key}"]
            )

            logger.info(f"Retrieved {len(available_players)} available players for league {league_key}")
            return available_players

        except Exception as e:
            logger.error(f"Error getting available players for league {league_key}: {e}")
            raise

    async def get_injury_report(self, league_key: str = None) -> List[Dict[str, Any]]:
        """
        Get current injury report for players.

        Args:
            league_key: Optional league context for relevant players

        Returns:
            List of injury report dictionaries
        """
        cache_key = f"injury_report:{league_key or 'all'}"

        # Check cache first
        cached_data = await self.cache_manager.get(cache_key)
        if cached_data is not None:
            logger.debug("Returning cached injury report")
            return cached_data

        try:
            # This would typically call a specialized injury report endpoint
            # For now, we'll get it through available players with injury status
            available_players = await self.get_available_players(
                league_key,
                status="A",  # All players to check injury status
                count=500
            )

            # Filter for injured players
            injured_players = []
            for player in available_players:
                if player.get('injury_status') and player['injury_status'] != 'Healthy':
                    injury_info = {
                        'player_key': player['player_key'],
                        'player_name': player['name'],
                        'team': player.get('team'),
                        'position': player.get('position'),
                        'injury_status': player['injury_status'],
                        'injury_note': player.get('injury_note', ''),
                        'last_updated': datetime.utcnow().isoformat()
                    }
                    injured_players.append(injury_info)

            # Cache injury report with medium TTL
            await self.cache_manager.set(
                cache_key,
                injured_players,
                ttl=timedelta(hours=2),
                tags=["injury_report", "yahoo_api"]
            )

            logger.info(f"Retrieved injury report with {len(injured_players)} injured players")
            return injured_players

        except Exception as e:
            logger.error(f"Error getting injury report: {e}")
            raise

    async def get_opponent_roster(
        self,
        league_key: str,
        opponent_team_key: str,
        week: int = None
    ) -> Dict[str, Any]:
        """
        Get opponent team roster for matchup analysis.

        Args:
            league_key: Yahoo league identifier
            opponent_team_key: Yahoo opponent team identifier
            week: Optional week number (current week if not specified)

        Returns:
            Opponent roster information dictionary
        """
        cache_key = f"opponent_roster:{league_key}:{opponent_team_key}:{week or 'current'}"

        # Check cache first
        cached_data = await self.cache_manager.get(cache_key)
        if cached_data is not None:
            logger.debug(f"Returning cached opponent roster for team {opponent_team_key}, week {week}")
            return cached_data

        try:
            # Use the existing get_roster method with opponent team key
            roster_info = await self.get_roster(league_key, opponent_team_key, week)

            # Add opponent-specific metadata
            roster_info['is_opponent'] = True
            roster_info['opponent_team_key'] = opponent_team_key

            # Cache opponent roster data
            await self.cache_manager.set(
                cache_key,
                roster_info,
                ttl=timedelta(hours=2),  # Same TTL as regular rosters
                tags=["roster", "opponent", "yahoo_api", f"league:{league_key}"]
            )

            logger.info(f"Retrieved opponent roster for team {opponent_team_key}, {len(roster_info['players'])} players")
            return roster_info

        except Exception as e:
            logger.error(f"Error getting opponent roster for team {opponent_team_key}: {e}")
            raise

    async def fetch_multiple_leagues_data(
        self,
        league_keys: List[str],
        data_types: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch data for multiple leagues in parallel.

        Args:
            league_keys: List of Yahoo league identifiers
            data_types: List of data types to fetch (roster, matchup, standings, etc.)

        Returns:
            Dictionary mapping league_key to fetched data
        """
        if data_types is None:
            data_types = ["roster", "standings"]

        logger.info(f"Fetching data for {len(league_keys)} leagues in parallel")

        # Create tasks for parallel execution
        tasks = []
        for league_key in league_keys:
            task = asyncio.create_task(
                self._fetch_league_data(league_key, data_types),
                name=f"fetch_league_{league_key}"
            )
            tasks.append(task)

        # Execute tasks with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.settings.async_timeout_seconds * len(league_keys)
            )

            # Process results
            league_data = {}
            for i, result in enumerate(results):
                league_key = league_keys[i]
                if isinstance(result, Exception):
                    logger.error(f"Error fetching data for league {league_key}: {result}")
                    league_data[league_key] = {"error": str(result)}
                else:
                    league_data[league_key] = result

            logger.info(f"Completed parallel fetch for {len(league_keys)} leagues")
            return league_data

        except asyncio.TimeoutError:
            logger.error("Timeout while fetching multiple leagues data")
            raise
        except Exception as e:
            logger.error(f"Error in parallel league data fetch: {e}")
            raise

    async def _fetch_league_data(self, league_key: str, data_types: List[str]) -> Dict[str, Any]:
        """Fetch specific data types for a single league."""
        league_data = {"league_key": league_key}

        # Fetch each requested data type
        for data_type in data_types:
            try:
                if data_type == "roster":
                    # Get roster for the user's team (assuming first team)
                    # This would need team identification logic in a real implementation
                    pass
                elif data_type == "standings":
                    # Implementation for standings
                    pass
                elif data_type == "available_players":
                    league_data["available_players"] = await self.get_available_players(league_key)

            except Exception as e:
                logger.error(f"Error fetching {data_type} for league {league_key}: {e}")
                league_data[data_type] = {"error": str(e)}

        return league_data

    async def get_league_wide_ownership_data(self, player_keys: List[str], league_key: str) -> Dict[str, Dict[str, Any]]:
        """Get league-wide ownership percentage for specific players."""
        try:
            self._yahoo_client.league_id = league_key.split(".")[-1]

            ownership_data = {}
            logger.info(f"Getting league-wide ownership data for {len(player_keys)} players")

            # Try different approaches to get ownership data (prioritizing CORRECT yfpy methods)
            approaches = [
                ("team roster with ownership (CORRECT)", self._get_team_roster_with_ownership),
                ("individual player percent_owned (CORRECT)", self._get_individual_player_ownership),
                ("get_league_players with ownership", self._get_league_players_with_ownership),
                ("fixed league players approach", self._get_league_players_fixed)
            ]

            for approach_name, approach_func in approaches:
                try:
                    logger.info(f"Trying approach: {approach_name}")
                    ownership_data = await approach_func(player_keys)
                    if ownership_data:
                        logger.info(f"✅ Success with {approach_name}: Retrieved data for {len(ownership_data)} players")
                        break
                    else:
                        logger.warning(f"❌ {approach_name} returned no data")
                except Exception as e:
                    logger.warning(f"❌ {approach_name} failed: {e}")
                    continue

            if not ownership_data:
                logger.warning("⚠️  All ownership data approaches failed, using empty data")

            return ownership_data

        except Exception as e:
            logger.error(f"Error getting league-wide ownership data: {e}")
            return {}

    async def _get_league_players_with_ownership(self, player_keys: List[str]) -> Dict[str, Dict[str, Any]]:
        """Try getting ownership data from league players endpoint."""
        ownership_data = {}

        league_players = self._yahoo_client.get_league_players()

        if league_players and hasattr(league_players, 'players'):
            for player in league_players.players:
                if hasattr(player, 'player_key') and player.player_key in player_keys:
                    player_ownership = self._extract_ownership_from_player(player)
                    if player_ownership:
                        ownership_data[player.player_key] = player_ownership

        return ownership_data

    async def _get_individual_player_ownership(self, player_keys: List[str]) -> Dict[str, Dict[str, Any]]:
        """Try getting ownership data from individual player queries using CORRECT yfpy methods."""
        ownership_data = {}

        for player_key in player_keys[:5]:  # Limit to 5 to avoid rate limits
            try:
                # Use the ACTUAL yfpy method for getting percent owned
                player = self._yahoo_client.get_player_percent_owned_by_week(player_key, chosen_week="current")
                if player:
                    player_ownership = self._extract_ownership_from_player(player)
                    if player_ownership:
                        ownership_data[player_key] = player_ownership
                        logger.info(f"✅ Got ownership for {player_key}: {player_ownership}")

            except Exception as e:
                logger.debug(f"Player percent owned query failed for {player_key}: {e}")

                # Try alternative method
                try:
                    player = self._yahoo_client.get_player_ownership(player_key)
                    if player:
                        player_ownership = self._extract_ownership_from_player(player)
                        if player_ownership:
                            ownership_data[player_key] = player_ownership
                            logger.info(f"✅ Got ownership via alternative for {player_key}: {player_ownership}")
                except Exception as e2:
                    logger.debug(f"Player ownership query also failed for {player_key}: {e2}")
                    continue

        return ownership_data

    async def _get_team_roster_with_ownership(self, player_keys: List[str]) -> Dict[str, Dict[str, Any]]:
        """Try getting ownership data from team roster info which includes ownership data."""
        ownership_data = {}

        try:
            # Get current user's team roster with full player info including ownership
            user_team_key = await self.get_user_team_key(f"461.l.{self._yahoo_client.league_id}")
            team_id = user_team_key.split('.')[-1]

            # Use the method that includes ownership data
            roster_players = self._yahoo_client.get_team_roster_player_info_by_week(
                team_id=int(team_id),
                chosen_week="current"
            )

            if roster_players:
                for player in roster_players:
                    if hasattr(player, 'player_key') and player.player_key in player_keys:
                        player_ownership = self._extract_ownership_from_player(player)
                        if player_ownership:
                            ownership_data[player.player_key] = player_ownership
                            logger.info(f"✅ Got roster ownership for {player.player_key}: {player_ownership}")

        except Exception as e:
            logger.debug(f"Team roster ownership query failed: {e}")

        return ownership_data

    async def _get_league_players_fixed(self, player_keys: List[str]) -> Dict[str, Dict[str, Any]]:
        """Try getting ownership data from league players endpoint with fixes."""
        ownership_data = {}

        try:
            # Try different parameters for get_league_players
            for week in [None, 1]:  # Try current week and week 1
                try:
                    if week:
                        league_players = self._yahoo_client.get_league_players(week=week)
                    else:
                        league_players = self._yahoo_client.get_league_players()

                    if league_players and hasattr(league_players, 'players'):
                        for player in league_players.players:
                            if hasattr(player, 'player_key') and player.player_key in player_keys:
                                player_ownership = self._extract_ownership_from_player(player)
                                if player_ownership:
                                    ownership_data[player.player_key] = player_ownership

                        if ownership_data:  # Found some data, break
                            logger.info(f"League players method worked with week={week}")
                            break

                except Exception as e:
                    logger.debug(f"League players with week={week} failed: {e}")
                    continue
        except Exception as e:
            logger.debug(f"League players approaches failed: {e}")

        return ownership_data

    def _extract_ownership_from_player(self, player) -> Dict[str, Any]:
        """Extract ownership data from a player object."""
        player_ownership = {}

        # Get percent owned across all Yahoo leagues (% ROS)
        if hasattr(player, 'percent_owned') and player.percent_owned:
            if hasattr(player.percent_owned, 'value'):
                player_ownership['percent_owned'] = float(player.percent_owned.value)
            elif hasattr(player.percent_owned, 'coverage_value'):
                player_ownership['percent_owned'] = float(player.percent_owned.coverage_value)
            elif hasattr(player.percent_owned, 'percent_owned_value'):
                player_ownership['percent_owned'] = float(player.percent_owned.percent_owned_value)
            else:
                try:
                    player_ownership['percent_owned'] = float(str(player.percent_owned))
                except (ValueError, TypeError):
                    pass

        # Get percent started across all Yahoo leagues (% Start) - what the mob is actually doing!
        start_attrs = [
            'percent_started', 'start_percentage', 'started_percentage',
            'start_percent', 'percent_start', 'started_pct'
        ]

        for attr_name in start_attrs:
            if hasattr(player, attr_name):
                percent_started_attr = getattr(player, attr_name)
                if percent_started_attr:
                    try:
                        if hasattr(percent_started_attr, 'value'):
                            player_ownership['percent_started'] = float(percent_started_attr.value)
                        elif hasattr(percent_started_attr, 'coverage_value'):
                            player_ownership['percent_started'] = float(percent_started_attr.coverage_value)
                        elif hasattr(percent_started_attr, 'percent_started_value'):
                            player_ownership['percent_started'] = float(percent_started_attr.percent_started_value)
                        else:
                            player_ownership['percent_started'] = float(str(percent_started_attr))
                        logger.debug(f"Found start percentage via {attr_name}: {player_ownership['percent_started']}")
                        break  # Found it, stop looking
                    except (ValueError, TypeError):
                        continue

        # Also check if player has a stats object with start data
        if hasattr(player, 'player_stats') and player.player_stats:
            stats = player.player_stats
            if hasattr(stats, 'percent_started'):
                try:
                    player_ownership['percent_started'] = float(stats.percent_started)
                    logger.debug(f"Found start percentage in player_stats: {player_ownership['percent_started']}")
                except (ValueError, TypeError):
                    pass

        # Get any other ownership metrics
        if hasattr(player, 'ownership'):
            if hasattr(player.ownership, 'ownership_type'):
                player_ownership['ownership_type'] = player.ownership.ownership_type

        return player_ownership

    async def _initialize_yahoo_client(self) -> None:
        """Initialize Yahoo Fantasy Sports API client."""
        try:
            # Create Yahoo API client with OAuth2 credentials
            from pathlib import Path

            # First, create a temporary client to get the correct game_key for 2025
            temp_client = YahooFantasySportsQuery(
                league_id=None,
                game_code="nfl",
                yahoo_consumer_key=self.settings.yahoo_client_id,
                yahoo_consumer_secret=self.settings.yahoo_client_secret,
                env_file_location=Path(".")
            )

            # Get the proper game_key for 2025 NFL season to stop the warnings
            try:
                game_key_2025 = temp_client.get_game_key_by_season(2025)
                logger.info(f"Retrieved game_key for 2025 NFL season: {game_key_2025}")
            except Exception as e:
                logger.warning(f"Could not get 2025 game_key, using default: {e}")
                game_key_2025 = "461"  # Fallback to current game_key from logs

            # Create the main client with the correct game_id
            self._yahoo_client = YahooFantasySportsQuery(
                league_id=None,  # Will be set per request
                game_code="nfl",
                game_id=game_key_2025,  # Set the correct game_key to stop warnings
                yahoo_consumer_key=self.settings.yahoo_client_id,
                yahoo_consumer_secret=self.settings.yahoo_client_secret,
                env_file_location=Path(".")  # OAuth tokens stored in current directory
            )

            logger.info("Yahoo API client initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Yahoo API client: {e}")
            raise AuthenticationError(f"Yahoo API authentication failed: {e}")

    async def _make_api_request(self, request: APIRequest) -> Any:
        """
        Make API request with rate limiting, retry logic, and error handling.

        Args:
            request: API request configuration

        Returns:
            API response data
        """
        async with self._semaphore:
            # Check rate limits
            if not self.rate_limiter.can_make_request():
                wait_time = self.rate_limiter.time_until_reset().total_seconds()
                logger.warning(f"Rate limit exceeded, waiting {wait_time} seconds")
                if wait_time > 0:
                    await asyncio.sleep(min(wait_time, 300))  # Max 5 minute wait

                if not self.rate_limiter.can_make_request():
                    raise RateLimitError("API rate limit exceeded")

            # Retry logic
            last_exception = None
            for attempt in range(request.max_retries + 1):
                try:
                    # Calculate backoff delay
                    if attempt > 0:
                        delay = request.backoff_factor ** attempt
                        logger.debug(f"Retrying request after {delay}s delay (attempt {attempt + 1})")
                        await asyncio.sleep(delay)

                    # Make the actual API call
                    response = await self._execute_yahoo_request(request)

                    # Record successful request
                    self.rate_limiter.record_request()

                    logger.debug(f"API request successful: {request.endpoint}")
                    return response

                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    last_exception = e
                    logger.warning(f"API request failed (attempt {attempt + 1}): {e}")

                    if attempt == request.max_retries:
                        break

                except RateLimitError:
                    # Don't retry rate limit errors immediately
                    raise
                except Exception as e:
                    logger.error(f"Unexpected error in API request: {e}")
                    raise

            # All retries exhausted
            logger.error(f"API request failed after {request.max_retries + 1} attempts")
            raise last_exception or Exception("API request failed")

    async def _execute_yahoo_request(self, request: APIRequest) -> Any:
        """Execute the actual Yahoo API request."""
        try:
            if not self._yahoo_client:
                await self._initialize_yahoo_client()

            # Route to appropriate Yahoo API method
            if request.endpoint == APIEndpoint.USER_LEAGUES:
                game_key = request.params.get("game_key", "nfl")
                return self._yahoo_client.get_user_leagues_by_game_key(game_key)

            elif request.endpoint == APIEndpoint.TEAM_ROSTER:
                league_key = request.params["league_key"]
                team_key = request.params["team_key"]
                week = request.params.get("week")

                # Set league context
                self._yahoo_client.league_id = league_key.split(".")[-1]

                if week:
                    return self._yahoo_client.get_team_roster_by_week(
                        team_id=team_key.split(".")[-1],
                        chosen_week=week
                    )
                else:
                    # For current week, we still need to use get_team_roster_by_week with current week
                    # since there's no simple get_team_roster method
                    return self._yahoo_client.get_team_roster_by_week(
                        team_id=team_key.split(".")[-1],
                        chosen_week=1  # Default to week 1 if no week specified
                    )

            elif request.endpoint == APIEndpoint.TEAM_MATCHUP:
                league_key = request.params["league_key"]
                team_key = request.params["team_key"]
                week = request.params["week"]

                self._yahoo_client.league_id = league_key.split(".")[-1]
                return self._yahoo_client.get_team_matchups(
                    team_id=team_key.split(".")[-1]
                )

            elif request.endpoint == APIEndpoint.PLAYER_INFO:
                player_key = request.params["player_key"]
                return self._yahoo_client.get_player_info(player_key)

            elif request.endpoint == APIEndpoint.AVAILABLE_PLAYERS:
                league_key = request.params["league_key"]
                league_id = league_key.split(".")[-1]
                self._yahoo_client.league_id = league_id

                # Debug: Verify league ID is set correctly
                logger.debug(f"Set league_id to: {self._yahoo_client.league_id}")
                logger.debug(f"League key: {league_key}, extracted ID: {league_id}")

                # Use basic get_league_players with only supported parameters
                # Note: yfpy may not support filtering, so we'll filter after retrieval
                return self._yahoo_client.get_league_players(
                    player_count_limit=request.params.get("count", 25),
                    player_count_start=0
                )

            else:
                raise ValueError(f"Unsupported endpoint: {request.endpoint}")

        except Exception as e:
            logger.error(f"Yahoo API request execution failed: {e}")
            raise

    async def _transform_yahoo_player(self, yahoo_player: YfpyPlayer) -> Dict[str, Any]:
        """
        Transform Yahoo player object to our internal format.

        Args:
            yahoo_player: Yahoo API player object

        Returns:
            Player information dictionary in our format
        """
        try:
            # Map Yahoo position to our Position enum
            position_map = {
                "QB": Position.QB,
                "RB": Position.RB,
                "WR": Position.WR,
                "TE": Position.TE,
                "K": Position.K,
                "DEF": Position.DEF
            }

            # Map Yahoo team to our Team enum
            yahoo_team = getattr(yahoo_player, 'editorial_team_abbr', '') or getattr(yahoo_player, 'team_abbr', '')
            nfl_team = None
            try:
                nfl_team = NFLTeam(yahoo_team.upper()) if yahoo_team else None
            except ValueError:
                logger.warning(f"Unknown NFL team: {yahoo_team}")

            # Debug Yahoo player object
            logger.debug(f"Transforming Yahoo player: {yahoo_player.player_key}")
            logger.debug(f"Player name object: {getattr(yahoo_player, 'name', 'NO NAME ATTR')}")
            logger.debug(f"Player primary_position: {getattr(yahoo_player, 'primary_position', 'NO POSITION ATTR')}")
            logger.debug(f"Player team: {yahoo_team}")

            # Get name safely
            name = "Unknown Player"
            if hasattr(yahoo_player, 'name') and yahoo_player.name:
                if hasattr(yahoo_player.name, 'first') and hasattr(yahoo_player.name, 'last'):
                    name = f"{yahoo_player.name.first} {yahoo_player.name.last}"
                elif hasattr(yahoo_player.name, 'full'):
                    name = yahoo_player.name.full
                else:
                    name = str(yahoo_player.name)
            elif hasattr(yahoo_player, 'full_name'):
                name = yahoo_player.full_name

            # Get position safely
            position = getattr(yahoo_player, 'primary_position', '')
            if not position:
                position = getattr(yahoo_player, 'position', '')
            if not position:
                position = getattr(yahoo_player, 'display_position', '')

            logger.info(f"Extracted player info - Name: '{name}', Position: '{position}', Team: '{yahoo_team}'")

            # Extract detailed ownership and availability data
            ownership_data = {}
            if hasattr(yahoo_player, 'percent_owned') and yahoo_player.percent_owned:
                # Get percentage owned across all Yahoo leagues (% ROS - Roster %)
                if hasattr(yahoo_player.percent_owned, 'value'):
                    ownership_data['percent_owned'] = float(yahoo_player.percent_owned.value)
                elif hasattr(yahoo_player.percent_owned, 'coverage_value'):
                    ownership_data['percent_owned'] = float(yahoo_player.percent_owned.coverage_value)
                else:
                    ownership_data['percent_owned'] = float(str(yahoo_player.percent_owned))

            # Extract start percentage data (% Start) - what the mob is actually doing!
            if hasattr(yahoo_player, 'percent_started') and yahoo_player.percent_started:
                if hasattr(yahoo_player.percent_started, 'value'):
                    ownership_data['percent_started'] = float(yahoo_player.percent_started.value)
                elif hasattr(yahoo_player.percent_started, 'coverage_value'):
                    ownership_data['percent_started'] = float(yahoo_player.percent_started.coverage_value)
                else:
                    ownership_data['percent_started'] = float(str(yahoo_player.percent_started))

            # Also check for start percentage under different attribute names
            for attr_name in ['percent_started', 'start_percentage', 'started_percentage']:
                if 'percent_started' not in ownership_data and hasattr(yahoo_player, attr_name):
                    percent_started_attr = getattr(yahoo_player, attr_name)
                    if percent_started_attr:
                        try:
                            if hasattr(percent_started_attr, 'value'):
                                ownership_data['percent_started'] = float(percent_started_attr.value)
                            elif hasattr(percent_started_attr, 'coverage_value'):
                                ownership_data['percent_started'] = float(percent_started_attr.coverage_value)
                            else:
                                ownership_data['percent_started'] = float(str(percent_started_attr))
                            break  # Found it, stop looking
                        except (ValueError, TypeError):
                            continue

            # Get ownership status (owned, available, waivers, etc.)
            ownership_status = 'available'
            if hasattr(yahoo_player, 'ownership') and yahoo_player.ownership:
                ownership_status = getattr(yahoo_player.ownership, 'ownership_type', 'available')
                # Also get ownership destination if owned by another team
                if hasattr(yahoo_player.ownership, 'destination_team_key'):
                    ownership_data['owned_by_team'] = yahoo_player.ownership.destination_team_key
                if hasattr(yahoo_player.ownership, 'destination_team_name'):
                    ownership_data['owned_by_team_name'] = yahoo_player.ownership.destination_team_name

            # Extract waiver/trade information
            if hasattr(yahoo_player, 'status_full'):
                ownership_data['status_full'] = yahoo_player.status_full
            if hasattr(yahoo_player, 'on_disabled_list'):
                ownership_data['on_disabled_list'] = yahoo_player.on_disabled_list

            # Basic player information
            player_info = {
                'player_key': yahoo_player.player_key,
                'name': name,
                'position': position,
                'team': yahoo_team,
                'season': 2025,  # Current NFL season
                'jersey_number': getattr(yahoo_player, 'jersey_number', None),
                'bye_weeks': getattr(yahoo_player, 'bye_weeks', []),
                'is_undroppable': getattr(yahoo_player, 'is_undroppable', False),
                'ownership_status': ownership_status,
                'ownership_data': ownership_data  # Enhanced ownership info
            }

            # Injury information
            if hasattr(yahoo_player, 'status') and yahoo_player.status:
                player_info['injury_status'] = yahoo_player.status
            else:
                player_info['injury_status'] = 'Healthy'

            if hasattr(yahoo_player, 'injury_note'):
                player_info['injury_note'] = yahoo_player.injury_note

            # Statistics if available
            if hasattr(yahoo_player, 'player_stats') and yahoo_player.player_stats:
                stats = {}
                for stat in yahoo_player.player_stats.stats:
                    # Handle different Yahoo stat object structures
                    if hasattr(stat, 'stat') and hasattr(stat.stat, 'display_name'):
                        stats[stat.stat.display_name] = stat.value
                    elif hasattr(stat, 'display_name'):
                        stats[stat.display_name] = getattr(stat, 'value', 0)
                    elif hasattr(stat, 'stat_id'):
                        stats[f'stat_{stat.stat_id}'] = getattr(stat, 'value', 0)
                player_info['stats'] = stats

            # Projected points if available
            if hasattr(yahoo_player, 'player_points') and yahoo_player.player_points:
                player_info['projected_points'] = yahoo_player.player_points.total

            # Enhanced player context data from Yahoo
            enhanced_context = {}

            # Get player rank if available
            if hasattr(yahoo_player, 'player_rank'):
                enhanced_context['player_rank'] = yahoo_player.player_rank

            # Get player notes (news, updates)
            if hasattr(yahoo_player, 'player_notes') and yahoo_player.player_notes:
                enhanced_context['player_notes'] = yahoo_player.player_notes

            # Get editorial rankings if available
            if hasattr(yahoo_player, 'editorial_rank'):
                enhanced_context['editorial_rank'] = yahoo_player.editorial_rank

            # Get position rank if available
            if hasattr(yahoo_player, 'position_rank'):
                enhanced_context['position_rank'] = yahoo_player.position_rank

            # Get player experience/years pro if available
            if hasattr(yahoo_player, 'years_pro'):
                enhanced_context['years_pro'] = yahoo_player.years_pro

            # Get player age if available
            if hasattr(yahoo_player, 'age'):
                enhanced_context['age'] = yahoo_player.age

            # Get uniform number
            if hasattr(yahoo_player, 'uniform_number'):
                enhanced_context['uniform_number'] = yahoo_player.uniform_number

            # Get depth chart info if available
            if hasattr(yahoo_player, 'depth_chart'):
                enhanced_context['depth_chart'] = yahoo_player.depth_chart

            # Add enhanced context if any data was found
            if enhanced_context:
                player_info['enhanced_context'] = enhanced_context

            return player_info

        except Exception as e:
            logger.error(f"Error transforming Yahoo player data: {e}")
            # Return minimal player info if transformation fails
            return {
                'player_key': getattr(yahoo_player, 'player_key', ''),
                'name': 'Unknown Player',
                'position': '',
                'team': '',
                'season': 2025,
                'error': str(e)
            }

    async def get_player_historical_stats(
        self,
        player_key: str,
        league_key: str,
        weeks_back: int = 5
    ) -> List[Dict[str, Any]]:
        """Get historical weekly stats for a player."""
        try:
            self._yahoo_client.league_id = league_key.split(".")[-1]

            historical_stats = []
            # Since it's early in season, try to get any available stats from recent weeks
            # Start from week 1 and try up to current week
            available_weeks = [1, 2, 3, 4, 5]  # Try first few weeks of season

            # Get stats for available weeks
            for week in available_weeks[:weeks_back]:
                try:
                    week_stats = self._yahoo_client.get_player_stats_by_week(
                        player_key, week
                    )

                    if week_stats and hasattr(week_stats, 'player_stats'):
                        stats_dict = {
                            'week': week,
                            'stats': {}
                        }

                        # Extract stats - Yahoo API structure: stat_id, value, display_name
                        for stat in week_stats.player_stats.stats:
                            # Get stat name/key
                            stat_name = None
                            if hasattr(stat, 'display_name') and stat.display_name:
                                stat_name = stat.display_name
                            elif hasattr(stat, 'display') and stat.display:
                                stat_name = stat.display
                            elif hasattr(stat, 'stat_id'):
                                stat_name = f'stat_{stat.stat_id}'

                            # Get stat value
                            stat_value = getattr(stat, 'value', 0)

                            if stat_name:
                                stats_dict['stats'][stat_name] = stat_value

                        historical_stats.append(stats_dict)

                except Exception as e:
                    logger.debug(f"Could not get week {week} stats for {player_key}: {e}")
                    continue

            return historical_stats

        except Exception as e:
            logger.error(f"Error getting historical stats for {player_key}: {e}")
            return []

    async def get_player_season_stats(
        self,
        player_key: str,
        league_key: str
    ) -> Dict[str, Any]:
        """Get full season stats for a player."""
        try:
            self._yahoo_client.league_id = league_key.split(".")[-1]

            season_stats = self._yahoo_client.get_player_stats_for_season(player_key)

            if season_stats and hasattr(season_stats, 'player_stats'):
                stats_dict = {}

                for stat in season_stats.player_stats.stats:
                    # Get stat name/key
                    stat_name = None
                    if hasattr(stat, 'display_name') and stat.display_name:
                        stat_name = stat.display_name
                    elif hasattr(stat, 'display') and stat.display:
                        stat_name = stat.display
                    elif hasattr(stat, 'stat_id'):
                        stat_name = f'stat_{stat.stat_id}'

                    # Get stat value
                    stat_value = getattr(stat, 'value', 0)

                    if stat_name:
                        stats_dict[stat_name] = stat_value

                return {
                    'player_key': player_key,
                    'season_stats': stats_dict
                }

            return {}

        except Exception as e:
            logger.debug(f"Could not get season stats for {player_key}: {e}")
            return {}

    async def get_opponent_defensive_stats(
        self,
        opponent_team: str,
        position: str,
        league_key: str
    ) -> Dict[str, Any]:
        """Get opponent defensive stats against a position."""
        try:
            # This would require league-wide analysis of how opponent performs vs position
            # For now, return placeholder that we can enhance
            return {
                'opponent_team': opponent_team,
                'position': position,
                'points_allowed_avg': 0.0,
                'rank_vs_position': 16,  # Middle of pack
                'recent_trend': 'neutral'
            }

        except Exception as e:
            logger.debug(f"Could not get defensive stats for {opponent_team} vs {position}: {e}")
            return {}

    async def get_league_matchups_by_week(
        self,
        league_key: str,
        week: int
    ) -> List[Dict[str, Any]]:
        """Get all league matchups for a specific week."""
        try:
            self._yahoo_client.league_id = league_key.split(".")[-1]

            matchups = self._yahoo_client.get_league_matchups_by_week(week)

            matchup_list = []
            if matchups:
                for matchup in matchups:
                    matchup_dict = {
                        'week': week,
                        'teams': [],
                        'projected_scores': []
                    }

                    # Extract team info from matchup
                    if hasattr(matchup, 'teams'):
                        for team in matchup.teams:
                            team_info = {
                                'team_key': getattr(team, 'team_key', ''),
                                'name': getattr(team, 'name', ''),
                                'projected_points': getattr(team, 'projected_points', 0)
                            }
                            matchup_dict['teams'].append(team_info)

                    matchup_list.append(matchup_dict)

            return matchup_list

        except Exception as e:
            logger.debug(f"Could not get league matchups for week {week}: {e}")
            return []

    def _generate_cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate consistent cache key from endpoint and parameters."""
        # Sort parameters for consistent key generation
        sorted_params = sorted(params.items())
        param_string = "&".join([f"{k}={v}" for k, v in sorted_params])

        # Create hash of the full request
        full_string = f"{endpoint}?{param_string}"
        return hashlib.md5(full_string.encode()).hexdigest()

    async def get_all_teams(self, league_key: str) -> List[Dict[str, Any]]:
        """Get all teams in the league with basic info."""
        try:
            await self._initialize_yahoo_client()

            # Set league context
            self._yahoo_client.league_id = league_key.split(".")[-1]

            # Use yahoo client to get all teams in league
            teams = self._yahoo_client.get_league_teams()

            teams_data = []
            for team in teams:
                team_info = {
                    'team_id': getattr(team, 'team_id', None),
                    'team_key': getattr(team, 'team_key', None),
                    'name': self._convert_name_to_string(getattr(team, 'name', None)),
                    'manager_name': getattr(team, 'manager_nickname', 'Unknown Manager'),
                    'wins': getattr(team, 'team_standings', {}).get('wins', 0) if hasattr(team, 'team_standings') else 0,
                    'losses': getattr(team, 'team_standings', {}).get('losses', 0) if hasattr(team, 'team_standings') else 0,
                    'ties': getattr(team, 'team_standings', {}).get('ties', 0) if hasattr(team, 'team_standings') else 0,
                    'win_percentage': getattr(team, 'team_standings', {}).get('percentage', 0.0) if hasattr(team, 'team_standings') else 0.0,
                    'points_for': float(getattr(team, 'team_points', {}).get('total', 0)) if hasattr(team, 'team_points') else 0.0,
                    'points_against': float(getattr(team, 'team_points', {}).get('total', 0)) if hasattr(team, 'team_points') else 0.0,
                }
                teams_data.append(team_info)

            logger.info(f"Retrieved {len(teams_data)} teams from league {league_key}")
            return teams_data

        except Exception as e:
            logger.error(f"Failed to get all teams: {e}")
            # Return generic fallback data for any league size
            fallback_teams = []
            # Default to 12 teams if no other info available
            num_teams = 12
            for i in range(1, num_teams + 1):
                fallback_teams.append({
                    'team_id': i,
                    'team_key': f"{league_key}.t.{i}",
                    'name': f'Team {i}',
                    'manager_name': 'Unknown Manager',
                    'wins': 0,
                    'losses': 0,
                    'ties': 0,
                    'win_percentage': 0.0,
                    'points_for': 0.0,
                    'points_against': 0.0,
                })
            return fallback_teams

    def _convert_name_to_string(self, name_obj) -> str:
        """Convert Yahoo name object to string."""
        if hasattr(name_obj, 'full'):
            return str(name_obj.full)
        elif isinstance(name_obj, str):
            return name_obj
        elif hasattr(name_obj, '__dict__'):
            # Try to extract readable name from object
            for attr in ['full', 'name', 'first_last', 'display_name']:
                if hasattr(name_obj, attr):
                    value = getattr(name_obj, attr)
                    if value:
                        return str(value)
        return str(name_obj) if name_obj else "Unknown"

    async def get_team_info(self, league_key: str, team_key: str) -> Dict[str, Any]:
        """Get detailed information about a specific team."""
        try:
            await self._initialize_yahoo_client()

            # Set league context
            self._yahoo_client.league_id = league_key.split(".")[-1]

            # Get team data from Yahoo API
            team = self._yahoo_client.get_team_info(team_key.split(".")[-1])

            team_info = {
                'team_id': getattr(team, 'team_id', None),
                'team_key': getattr(team, 'team_key', None),
                'name': self._convert_name_to_string(getattr(team, 'name', None)),
                'manager_name': getattr(team, 'manager_nickname', 'Unknown Manager'),
                'wins': getattr(team, 'team_standings', {}).get('wins', 0) if hasattr(team, 'team_standings') else 0,
                'losses': getattr(team, 'team_standings', {}).get('losses', 0) if hasattr(team, 'team_standings') else 0,
                'ties': getattr(team, 'team_standings', {}).get('ties', 0) if hasattr(team, 'team_standings') else 0,
                'win_percentage': getattr(team, 'team_standings', {}).get('percentage', 0.0) if hasattr(team, 'team_standings') else 0.0,
                'points_for': float(getattr(team, 'team_points', {}).get('total', 0)) if hasattr(team, 'team_points') else 0.0,
                'points_against': float(getattr(team, 'team_points', {}).get('total', 0)) if hasattr(team, 'team_points') else 0.0,
                'games_played': getattr(team, 'team_standings', {}).get('games_played', 1) if hasattr(team, 'team_standings') else 1,
            }

            logger.info(f"Retrieved team info for {team_key}")
            return team_info

        except Exception as e:
            logger.error(f"Failed to get team info for {team_key}: {e}")
            # Return fallback data
            team_id = team_key.split('.')[-1] if '.' in team_key else '1'
            return {
                'team_id': team_id,
                'team_key': team_key,
                'name': f'Team {team_id}',
                'manager_name': 'Unknown Manager',
                'wins': 0,
                'losses': 0,
                'ties': 0,
                'win_percentage': 0.0,
                'points_for': 0.0,
                'points_against': 0.0,
                'games_played': 1,
            }
