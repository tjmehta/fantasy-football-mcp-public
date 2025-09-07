"""
Statistical Analysis Agent for Fantasy Football.

This module provides sophisticated statistical analysis for NFL players and teams,
including performance projections, trend analysis, and advanced metrics calculation.
"""

import asyncio
import statistics
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from loguru import logger

from ..models.player import Player, PlayerStats, PlayerProjections, Position, Team
from ..models.matchup import Matchup, GameEnvironment, WeatherCondition


@dataclass
class StatisticalMetrics:
    """Advanced statistical metrics for player analysis."""

    # Volume metrics
    target_share: Optional[float] = None
    air_yards_share: Optional[float] = None
    red_zone_share: Optional[float] = None
    goal_line_share: Optional[float] = None
    snap_share: Optional[float] = None

    # Efficiency metrics
    yards_per_target: Optional[float] = None
    yards_per_carry: Optional[float] = None
    yards_per_touch: Optional[float] = None
    touchdown_efficiency: Optional[float] = None
    catch_rate: Optional[float] = None

    # Advanced metrics
    air_yards_per_game: Optional[float] = None
    yards_after_catch_per_reception: Optional[float] = None
    broken_tackles_per_touch: Optional[float] = None
    pressure_rate_against: Optional[float] = None

    # Consistency metrics
    coefficient_of_variation: Optional[float] = None
    boom_rate: Optional[float] = None  # Games > 20 fantasy points
    bust_rate: Optional[float] = None  # Games < 5 fantasy points

    # Situational metrics
    home_vs_away_split: Optional[Dict[str, float]] = None
    weather_splits: Optional[Dict[str, float]] = None
    opponent_strength_splits: Optional[Dict[str, float]] = None


@dataclass
class RegressionResults:
    """Results from regression analysis."""

    model_type: str
    r2_score: float
    mse: float
    rmse: float
    coefficients: Optional[Dict[str, float]] = None
    feature_importance: Optional[Dict[str, float]] = None
    cross_val_scores: Optional[List[float]] = None
    residuals: Optional[np.ndarray] = None
    predictions: Optional[np.ndarray] = None


@dataclass
class ProjectionRange:
    """Projection with confidence intervals."""

    projected_value: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    standard_error: float
    variance: float


@dataclass
class TrendAnalysis:
    """Time series trend analysis results."""

    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_strength: float  # 0-1 scale
    seasonal_component: Optional[List[float]] = None
    residual_component: Optional[List[float]] = None
    autocorrelation: Optional[float] = None
    trend_regression_r2: float = 0.0
    changepoints: Optional[List[int]] = None


@dataclass
class WaiverAnalysis:
    """Analysis for waiver wire value."""

    player_id: str
    current_roster_percentage: float
    projected_ros_value: float
    breakout_probability: float
    injury_replacement_value: float
    schedule_strength: float
    opportunity_score: float
    recommendation: str  # "add", "hold", "drop", "monitor"
    confidence: float


class StatisticalAnalysisAgent:
    """
    Advanced statistical analysis agent for fantasy football players and teams.

    Provides sophisticated analysis including:
    - Multi-model performance projections
    - Advanced metric calculations
    - Regression analysis and trend detection
    - Matchup-adjusted projections
    - Confidence intervals and variance analysis
    """

    def __init__(self, max_workers: int = 4):
        """Initialize the statistical analysis agent."""
        self.max_workers = max_workers
        self.thread_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=max_workers)
        self.models_cache: Dict[str, Any] = {}

        # Suppress sklearn warnings
        warnings.filterwarnings("ignore", category=FutureWarning)

        logger.info(f"StatisticalAnalysisAgent initialized with {max_workers} workers")

    def _analyze_defensive_matchup(
        self,
        player: Player,
        opponent_team: str,
        historical_games: List[PlayerStats]
    ) -> Dict[str, Any]:
        """
        Analyze defensive matchup difficulty based on historical data.

        Args:
            player: Player to analyze
            opponent_team: Opposing team abbreviation
            historical_games: Player's historical stats

        Returns:
            Dictionary with matchup analysis
        """
        # Team defensive difficulty ratings (heuristic-based for Week 1)
        # These would normally come from API data but using smart defaults
        defensive_ratings = {
            # Elite pass defenses (harder matchups for QB/WR/TE)
            'BUF': {'pass_def': 'elite', 'run_def': 'average', 'overall': 'hard'},
            'PIT': {'pass_def': 'elite', 'run_def': 'good', 'overall': 'hard'},
            'NYJ': {'pass_def': 'elite', 'run_def': 'average', 'overall': 'hard'},
            'SF': {'pass_def': 'good', 'run_def': 'elite', 'overall': 'hard'},

            # Elite run defenses (harder for RB)
            'CLE': {'pass_def': 'average', 'run_def': 'elite', 'overall': 'hard'},
            'BAL': {'pass_def': 'good', 'run_def': 'elite', 'overall': 'hard'},

            # Weaker defenses (easier matchups)
            'CAR': {'pass_def': 'weak', 'run_def': 'weak', 'overall': 'easy'},
            'NYG': {'pass_def': 'weak', 'run_def': 'average', 'overall': 'easy'},
            'NE': {'pass_def': 'average', 'run_def': 'weak', 'overall': 'easy'},
            'DEN': {'pass_def': 'average', 'run_def': 'weak', 'overall': 'easy'},

            # Default for unknown teams
            'DEFAULT': {'pass_def': 'average', 'run_def': 'average', 'overall': 'average'}
        }

        # Get defensive rating for opponent
        def_rating = defensive_ratings.get(opponent_team.upper(), defensive_ratings['DEFAULT'])

        # Determine relevant defensive strength based on player position
        if player.position in ['QB', 'WR', 'TE']:
            relevant_def = def_rating['pass_def']
        elif player.position == 'RB':
            relevant_def = def_rating['run_def']
        else:
            relevant_def = def_rating['overall']

        # Calculate matchup multiplier
        multipliers = {
            'elite': 0.80,  # 20% harder matchup
            'good': 0.90,   # 10% harder
            'average': 1.00, # Neutral
            'weak': 1.15    # 15% easier matchup
        }

        matchup_multiplier = multipliers.get(relevant_def, 1.00)

        # Look for historical performance vs this opponent
        vs_opponent_games = [
            game for game in historical_games
            if hasattr(game, 'opponent') and game.opponent == opponent_team
        ]

        avg_vs_opponent = None
        if vs_opponent_games:
            avg_vs_opponent = sum(game.fantasy_points for game in vs_opponent_games) / len(vs_opponent_games)

        return {
            'opponent': opponent_team,
            'defensive_strength': relevant_def,
            'matchup_multiplier': matchup_multiplier,
            'difficulty': def_rating['overall'],
            'games_vs_opponent': len(vs_opponent_games),
            'avg_points_vs_opponent': avg_vs_opponent,
            'matchup_notes': self._get_matchup_notes(player.position, relevant_def, opponent_team)
        }

    def _get_matchup_notes(self, position: str, def_strength: str, opponent: str) -> str:
        """Generate human-readable matchup notes."""
        if def_strength == 'elite':
            return f"Tough matchup vs {opponent}'s elite {position} defense. Consider alternatives."
        elif def_strength == 'good':
            return f"Solid {opponent} defense. {position} may struggle for ceiling plays."
        elif def_strength == 'weak':
            return f"Smash spot vs {opponent}'s weak {position} defense. High upside play."
        else:
            return f"Neutral matchup vs {opponent}. Standard expectations."

    async def analyze_player(
        self,
        player: Player,
        historical_games: List[PlayerStats],
        upcoming_matchups: List[Matchup],
        league_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive statistical analysis of a player.

        Args:
            player: Player to analyze
            historical_games: List of historical game stats
            upcoming_matchups: Future matchups
            league_context: Additional league context data

        Returns:
            Dictionary containing comprehensive analysis results
        """
        logger.info(f"Starting comprehensive analysis for {player.name}")

        try:
            # Run analysis components in parallel
            tasks = [
                self._calculate_advanced_metrics(player, historical_games),
                self._analyze_performance_trends(historical_games),
                self._generate_projection_models(player, historical_games),
                self._analyze_matchup_difficulty(player, upcoming_matchups),
                self._calculate_situational_splits(player, historical_games)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            advanced_metrics = results[0] if not isinstance(results[0], Exception) else StatisticalMetrics()
            trend_analysis = results[1] if not isinstance(results[1], Exception) else None
            projection_models = results[2] if not isinstance(results[2], Exception) else {}
            matchup_analysis = results[3] if not isinstance(results[3], Exception) else {}
            situational_splits = results[4] if not isinstance(results[4], Exception) else {}

            # Generate final projections with confidence intervals
            ros_projections = await self._generate_ros_projections(
                player, historical_games, upcoming_matchups, projection_models
            )

            analysis_result = {
                "player_id": player.id,
                "player_name": player.name,
                "position": str(player.position),
                "team": str(player.team),
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "advanced_metrics": advanced_metrics,
                "trend_analysis": trend_analysis,
                "projection_models": projection_models,
                "matchup_analysis": matchup_analysis,
                "situational_splits": situational_splits,
                "ros_projections": ros_projections,
                "confidence_score": self._calculate_analysis_confidence(
                    len(historical_games), projection_models, trend_analysis
                )
            }

            logger.info(f"Completed analysis for {player.name}")
            return analysis_result

        except Exception as e:
            logger.error(f"Error analyzing player {player.name}: {str(e)}")
            raise

    async def analyze_team(
        self,
        team: Team,
        historical_stats: List[Dict[str, Any]],
        upcoming_matchups: List[Matchup]
    ) -> Dict[str, Any]:
        """
        Analyze team-level statistics and trends.

        Args:
            team: Team to analyze
            historical_stats: Historical team statistics
            upcoming_matchups: Upcoming team matchups

        Returns:
            Team analysis results
        """
        logger.info(f"Analyzing team: {str(team)}")

        try:
            # Convert to DataFrame for analysis
            df = pd.DataFrame(historical_stats)

            if df.empty:
                return {"team": str(team), "error": "No historical data available"}

            # Team-level metrics
            offensive_efficiency = self._calculate_offensive_efficiency(df)
            defensive_efficiency = self._calculate_defensive_efficiency(df)
            pace_metrics = self._calculate_pace_metrics(df)

            # Trend analysis
            scoring_trend = self._analyze_scoring_trend(df)
            defensive_trend = self._analyze_defensive_trend(df)

            # Strength of schedule analysis
            sos_analysis = await self._analyze_strength_of_schedule(team, upcoming_matchups)

            return {
                "team": str(team),
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "offensive_efficiency": offensive_efficiency,
                "defensive_efficiency": defensive_efficiency,
                "pace_metrics": pace_metrics,
                "scoring_trend": scoring_trend,
                "defensive_trend": defensive_trend,
                "strength_of_schedule": sos_analysis,
                "upcoming_matchups": len(upcoming_matchups)
            }

        except Exception as e:
            logger.error(f"Error analyzing team {str(team)}: {str(e)}")
            raise

    async def get_ros_projection(
        self,
        player: Player,
        historical_games: List[PlayerStats],
        remaining_schedule: List[Matchup]
    ) -> ProjectionRange:
        """
        Generate rest-of-season projection with confidence intervals.

        Args:
            player: Player to project
            historical_games: Historical performance data
            remaining_schedule: Remaining games

        Returns:
            ProjectionRange with confidence intervals
        """
        logger.info(f"Generating ROS projection for {player.name}")

        try:
            # Generate multiple projection models
            projection_models = await self._generate_projection_models(player, historical_games)

            # Calculate game-by-game projections
            game_projections = []
            for matchup in remaining_schedule:
                game_proj = await self._project_single_game(
                    player, historical_games, matchup, projection_models
                )
                game_projections.append(game_proj)

            # Aggregate to season total
            total_projection = sum(game_projections)

            # Calculate variance and confidence intervals
            variance = self._calculate_projection_variance(game_projections, projection_models)
            std_error = np.sqrt(variance)

            # 95% confidence interval
            confidence_level = 0.95
            z_score = norm.ppf((1 + confidence_level) / 2)
            margin_error = z_score * std_error

            return ProjectionRange(
                projected_value=total_projection,
                lower_bound=max(0, total_projection - margin_error),
                upper_bound=total_projection + margin_error,
                confidence_level=confidence_level,
                standard_error=std_error,
                variance=variance
            )

        except Exception as e:
            logger.error(f"Error generating ROS projection for {player.name}: {str(e)}")
            raise

    async def analyze_waiver_value(
        self,
        players: List[Player],
        historical_data: Dict[str, List[PlayerStats]],
        remaining_schedules: Dict[str, List[Matchup]],
        current_rosters: Dict[str, float]
    ) -> List[WaiverAnalysis]:
        """
        Analyze waiver wire value for multiple players.

        Args:
            players: List of players to analyze
            historical_data: Historical stats by player ID
            remaining_schedules: Remaining schedule by player ID
            current_rosters: Current roster percentage by player ID

        Returns:
            List of waiver analysis results
        """
        logger.info(f"Analyzing waiver value for {len(players)} players")

        try:
            # Analyze players in parallel
            tasks = []
            for player in players:
                task = self._analyze_single_waiver_candidate(
                    player,
                    historical_data.get(player.id, []),
                    remaining_schedules.get(player.id, []),
                    current_rosters.get(player.id, 0.0)
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions and sort by value
            waiver_analyses = [r for r in results if not isinstance(r, Exception)]
            waiver_analyses.sort(key=lambda x: x.projected_ros_value, reverse=True)

            return waiver_analyses

        except Exception as e:
            logger.error(f"Error analyzing waiver values: {str(e)}")
            raise

    async def analyze_multiple_players(
        self,
        players: List[Player],
        historical_data: Dict[str, List[PlayerStats]],
        upcoming_matchups: Dict[str, List[Matchup]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple players in parallel for efficiency.

        Args:
            players: List of players to analyze
            historical_data: Historical stats by player ID
            upcoming_matchups: Matchups by player ID

        Returns:
            List of analysis results
        """
        logger.info(f"Analyzing {len(players)} players in parallel")

        # Create analysis tasks
        tasks = []
        for player in players:
            task = self.analyze_player(
                player,
                historical_data.get(player.id, []),
                upcoming_matchups.get(player.id, [])
            )
            tasks.append(task)

        # Execute in parallel with controlled concurrency
        semaphore = asyncio.Semaphore(self.max_workers)

        async def bounded_analysis(task):
            async with semaphore:
                return await task

        bounded_tasks = [bounded_analysis(task) for task in tasks]
        results = await asyncio.gather(*bounded_tasks, return_exceptions=True)

        # Filter successful results
        successful_results = [r for r in results if not isinstance(r, Exception)]

        logger.info(f"Successfully analyzed {len(successful_results)} players")
        return successful_results

    # Private helper methods

    async def _calculate_advanced_metrics(
        self,
        player: Player,
        historical_games: List[PlayerStats]
    ) -> StatisticalMetrics:
        """Calculate advanced statistical metrics for a player."""
        if not historical_games:
            return StatisticalMetrics()

        try:
            # Convert to DataFrame for easier analysis
            df = self._stats_to_dataframe(historical_games)

            metrics = StatisticalMetrics()

            # Volume metrics
            if player.position in [Position.WR, Position.TE]:
                metrics.target_share = self._calculate_target_share(df)
                metrics.air_yards_share = self._calculate_air_yards_share(df)
                metrics.catch_rate = self._calculate_catch_rate(df)

            if player.position == Position.RB:
                metrics.red_zone_share = self._calculate_red_zone_share(df)
                metrics.goal_line_share = self._calculate_goal_line_share(df)

            # Efficiency metrics
            metrics.yards_per_target = self._safe_divide(
                df['receiving_yards'].sum(), df['targets'].sum()
            )
            metrics.yards_per_carry = self._safe_divide(
                df['rushing_yards'].sum(), df['rushing_attempts'].sum()
            )
            metrics.touchdown_efficiency = self._calculate_td_efficiency(df)

            # Consistency metrics
            fantasy_points = df['fantasy_points'].dropna()
            if len(fantasy_points) > 0:
                metrics.coefficient_of_variation = fantasy_points.std() / fantasy_points.mean()
                metrics.boom_rate = (fantasy_points >= 20).mean()
                metrics.bust_rate = (fantasy_points <= 5).mean()

            # Situational splits
            metrics.home_vs_away_split = self._calculate_home_away_splits(df)
            metrics.weather_splits = self._calculate_weather_splits(df)

            return metrics

        except Exception as e:
            logger.error(f"Error calculating advanced metrics: {str(e)}")
            return StatisticalMetrics()

    async def _analyze_performance_trends(
        self,
        historical_games: List[PlayerStats]
    ) -> Optional[TrendAnalysis]:
        """Analyze performance trends using time series analysis."""
        if len(historical_games) < 4:
            return None

        try:
            df = self._stats_to_dataframe(historical_games)
            fantasy_points = df['fantasy_points'].fillna(0)

            if len(fantasy_points) < 4:
                return None

            # Linear trend analysis
            x = np.arange(len(fantasy_points))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, fantasy_points)

            # Determine trend direction and strength
            if abs(r_value) < 0.1:
                trend_direction = "stable"
            elif slope > 0:
                trend_direction = "increasing"
            else:
                trend_direction = "decreasing"

            trend_strength = abs(r_value)

            # Seasonal decomposition if enough data
            seasonal_component = None
            residual_component = None

            if len(fantasy_points) >= 8:
                try:
                    decomposition = seasonal_decompose(
                        fantasy_points,
                        model='additive',
                        period=min(4, len(fantasy_points) // 2)
                    )
                    seasonal_component = decomposition.seasonal.tolist()
                    residual_component = decomposition.resid.dropna().tolist()
                except:
                    pass

            # Autocorrelation
            autocorr = None
            if len(fantasy_points) > 1:
                try:
                    autocorr = fantasy_points.autocorr(lag=1)
                except:
                    pass

            return TrendAnalysis(
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                seasonal_component=seasonal_component,
                residual_component=residual_component,
                autocorrelation=autocorr,
                trend_regression_r2=r_value**2
            )

        except Exception as e:
            logger.error(f"Error in trend analysis: {str(e)}")
            return None

    async def _generate_projection_models(
        self,
        player: Player,
        historical_games: List[PlayerStats]
    ) -> Dict[str, RegressionResults]:
        """Generate enhanced ML projection models for ensemble predictions."""
        if len(historical_games) < 3:
            return {}

        try:
            # Prepare feature matrix with enhanced features
            X, y = self._prepare_enhanced_modeling_data(player, historical_games)

            if X is None or y is None or len(X) < 3:
                return {}

            models = {}

            # Enhanced model suite
            models.update(await self._fit_enhanced_models(X, y, player))

            # Tree-based models
            if len(X) >= 5:  # Need more data for complex models
                models.update(await self._fit_tree_models(X, y))

            # Time series models
            ts_model = await self._fit_time_series_model(y)
            if ts_model:
                models['arima'] = ts_model

            return models

        except Exception as e:
            logger.error(f"Error generating projection models: {str(e)}")
            return {}

    def _prepare_enhanced_modeling_data(
        self,
        player: Player,
        historical_games: List[PlayerStats]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare enhanced feature matrix for ML models."""
        if len(historical_games) < 3:
            return None, None

        try:
            features = []
            targets = []

            for i, game in enumerate(historical_games):
                game_features = []

                # Basic stats
                game_features.extend([
                    getattr(game, 'passing_yards', 0),
                    getattr(game, 'rushing_yards', 0),
                    getattr(game, 'receiving_yards', 0),
                    getattr(game, 'passing_tds', 0),
                    getattr(game, 'rushing_tds', 0),
                    getattr(game, 'receiving_tds', 0),
                    getattr(game, 'receptions', 0),
                    getattr(game, 'targets', 0) if hasattr(game, 'targets') else 0,
                ])

                # Advanced features
                game_features.extend([
                    # Recency weight (more recent games weighted higher)
                    np.exp(-0.1 * (len(historical_games) - i - 1)),

                    # Game flow features
                    1 if getattr(game, 'home_game', True) else 0,  # Home/away
                    getattr(game, 'game_script', 0),  # Positive = ahead, negative = behind

                    # Efficiency metrics
                    getattr(game, 'snap_count', 0) / 70.0 if hasattr(game, 'snap_count') else 1.0,
                    getattr(game, 'target_share', 0) if hasattr(game, 'target_share') else 0,
                ])

                # Rolling averages (last 3 games)
                if i >= 2:
                    recent_games = historical_games[max(0, i-2):i+1]
                    avg_points = np.mean([g.fantasy_points for g in recent_games])
                    game_features.append(avg_points)
                else:
                    game_features.append(game.fantasy_points)

                features.append(game_features)
                targets.append(game.fantasy_points)

            return np.array(features), np.array(targets)

        except Exception as e:
            logger.error(f"Error preparing modeling data: {e}")
            return None, None

    async def _fit_enhanced_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        player: Player
    ) -> Dict[str, RegressionResults]:
        """Fit enhanced ML models including ensemble methods."""
        models = {}

        try:
            # Scale features for better model performance
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # 1. Weighted Linear Regression (emphasizes recent games)
            weights = np.exp(np.linspace(-1, 0, len(y)))  # Exponential decay
            lr_weighted = LinearRegression()
            lr_weighted.fit(X_scaled, y, sample_weight=weights)
            lr_pred = lr_weighted.predict(X_scaled)

            models['weighted_linear'] = RegressionResults(
                model_type='weighted_linear_regression',
                r2_score=r2_score(y, lr_pred),
                mse=mean_squared_error(y, lr_pred),
                rmse=np.sqrt(mean_squared_error(y, lr_pred)),
                coefficients={f'feature_{i}': coef for i, coef in enumerate(lr_weighted.coef_)},
                predictions=lr_pred
            )

            # 2. Ridge with optimal alpha (L2 regularization)
            alphas = [0.1, 1.0, 10.0, 100.0]
            best_alpha = 1.0
            best_score = -np.inf

            for alpha in alphas:
                ridge = Ridge(alpha=alpha)
                scores = cross_val_score(ridge, X_scaled, y, cv=min(3, len(y)-1), scoring='r2')
                avg_score = np.mean(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_alpha = alpha

            ridge_best = Ridge(alpha=best_alpha)
            ridge_best.fit(X_scaled, y)
            ridge_pred = ridge_best.predict(X_scaled)

            models['optimized_ridge'] = RegressionResults(
                model_type='optimized_ridge_regression',
                r2_score=r2_score(y, ridge_pred),
                mse=mean_squared_error(y, ridge_pred),
                rmse=np.sqrt(mean_squared_error(y, ridge_pred)),
                coefficients={f'feature_{i}': coef for i, coef in enumerate(ridge_best.coef_)},
                predictions=ridge_pred
            )

            # 3. Position-specific polynomial features
            if player.position in ['QB', 'RB']:  # More complex models for skill positions
                poly_features = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
                X_poly = poly_features.fit_transform(X_scaled[:, :5])  # Use top 5 features to avoid overfitting

                if X_poly.shape[1] <= len(y):  # Avoid overfitting
                    lr_poly = LinearRegression()
                    lr_poly.fit(X_poly, y)
                    poly_pred = lr_poly.predict(X_poly)

                    models['polynomial'] = RegressionResults(
                        model_type='polynomial_regression',
                        r2_score=r2_score(y, poly_pred),
                        mse=mean_squared_error(y, poly_pred),
                        rmse=np.sqrt(mean_squared_error(y, poly_pred)),
                        coefficients={f'poly_feature_{i}': coef for i, coef in enumerate(lr_poly.coef_)},
                        predictions=poly_pred
                    )

            return models

        except Exception as e:
            logger.error(f"Error fitting enhanced models: {e}")
            return {}

    async def _fit_tree_models(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, RegressionResults]:
        """Fit tree-based models for non-linear patterns."""
        models = {}

        try:
            # Random Forest with position-specific tuning
            rf = RandomForestRegressor(
                n_estimators=50,  # Smaller for limited data
                max_depth=3,      # Prevent overfitting
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
            rf.fit(X, y)
            rf_pred = rf.predict(X)

            models['random_forest'] = RegressionResults(
                model_type='random_forest',
                r2_score=r2_score(y, rf_pred),
                mse=mean_squared_error(y, rf_pred),
                rmse=np.sqrt(mean_squared_error(y, rf_pred)),
                coefficients={f'feature_importance_{i}': imp for i, imp in enumerate(rf.feature_importances_)},
                predictions=rf_pred
            )

            # Gradient Boosting for complex patterns
            gb = GradientBoostingRegressor(
                n_estimators=30,   # Conservative to avoid overfitting
                max_depth=2,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
            gb.fit(X, y)
            gb_pred = gb.predict(X)

            models['gradient_boosting'] = RegressionResults(
                model_type='gradient_boosting',
                r2_score=r2_score(y, gb_pred),
                mse=mean_squared_error(y, gb_pred),
                rmse=np.sqrt(mean_squared_error(y, gb_pred)),
                coefficients={f'feature_importance_{i}': imp for i, imp in enumerate(gb.feature_importances_)},
                predictions=gb_pred
            )

            return models

        except Exception as e:
            logger.error(f"Error fitting tree models: {e}")
            return {}

    async def _analyze_matchup_difficulty(
        self,
        player: Player,
        upcoming_matchups: List[Matchup]
    ) -> Dict[str, Any]:
        """Enhanced matchup difficulty analysis with defensive ratings."""
        if not upcoming_matchups:
            return {}

        try:
            matchup_scores = []
            weather_concerns = []
            defensive_analyses = []

            for matchup in upcoming_matchups:
                # Get opponent team from matchup
                opponent_team = getattr(matchup, 'opponent_team', None)
                if not opponent_team and hasattr(matchup, 'teams'):
                    # Try to find opponent from teams list
                    for team_info in matchup.teams:
                        if hasattr(team_info, 'team_abbr') and team_info.team_abbr != player.team:
                            opponent_team = team_info.team_abbr
                            break

                if opponent_team:
                    # Use our enhanced defensive matchup analysis
                    def_analysis = self._analyze_defensive_matchup(player, opponent_team, [])
                    defensive_analyses.append(def_analysis)

                    # Use the matchup multiplier as our difficulty score
                    # Convert to 1-10 scale (1=easiest, 10=hardest)
                    multiplier = def_analysis['matchup_multiplier']
                    if multiplier >= 1.10:
                        difficulty_score = 3.0  # Easy matchup
                    elif multiplier >= 1.05:
                        difficulty_score = 4.0  # Slightly easy
                    elif multiplier >= 0.95:
                        difficulty_score = 5.0  # Neutral
                    elif multiplier >= 0.90:
                        difficulty_score = 7.0  # Hard
                    else:
                        difficulty_score = 8.5  # Very hard

                    matchup_scores.append(difficulty_score)
                else:
                    # Fallback to original calculation
                    difficulty_score = self._calculate_matchup_difficulty(player, matchup)
                    matchup_scores.append(difficulty_score)

                # Check for weather concerns
                if matchup.game_environment and matchup.game_environment.is_weather_concern():
                    weather_concerns.append(matchup.id)

            avg_difficulty = statistics.mean(matchup_scores) if matchup_scores else 5.0

            return {
                "average_matchup_difficulty": avg_difficulty,
                "matchup_scores": matchup_scores,
                "defensive_analyses": defensive_analyses,
                "weather_games": weather_concerns,
                "total_games": len(upcoming_matchups),
                "difficulty_summary": self._get_difficulty_summary(avg_difficulty)
            }

        except Exception as e:
            logger.error(f"Error analyzing matchup difficulty: {str(e)}")
            return {}

    def _get_difficulty_summary(self, avg_difficulty: float) -> str:
        """Convert numeric difficulty to human-readable summary."""
        if avg_difficulty <= 3.5:
            return "Favorable matchups - exploit these spots!"
        elif avg_difficulty <= 4.5:
            return "Slightly favorable - good start candidate"
        elif avg_difficulty <= 5.5:
            return "Neutral matchups - standard expectations"
        elif avg_difficulty <= 7.5:
            return "Challenging matchups - may struggle"
        else:
            return "Very tough matchups - consider alternatives"

    async def _calculate_situational_splits(
        self,
        player: Player,
        historical_games: List[PlayerStats]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate performance splits by situation."""
        if not historical_games:
            return {}

        try:
            df = self._stats_to_dataframe(historical_games)
            splits = {}

            # Home vs Away (would need additional data)
            # For now, return placeholder structure
            splits["home_away"] = {
                "home_avg": 0.0,
                "away_avg": 0.0,
                "home_games": 0,
                "away_games": 0
            }

            # Weather splits (would need weather data)
            splits["weather"] = {
                "good_weather_avg": 0.0,
                "bad_weather_avg": 0.0,
                "good_weather_games": 0,
                "bad_weather_games": 0
            }

            # Recent performance (last 3, 5, 10 games)
            fantasy_points = df['fantasy_points'].dropna()
            if len(fantasy_points) > 0:
                splits["recent_performance"] = {
                    "last_3_avg": fantasy_points.tail(3).mean(),
                    "last_5_avg": fantasy_points.tail(5).mean(),
                    "last_10_avg": fantasy_points.tail(10).mean(),
                    "season_avg": fantasy_points.mean()
                }

            return splits

        except Exception as e:
            logger.error(f"Error calculating situational splits: {str(e)}")
            return {}

    async def _generate_ros_projections(
        self,
        player: Player,
        historical_games: List[PlayerStats],
        upcoming_matchups: List[Matchup],
        projection_models: Dict[str, RegressionResults]
    ) -> Dict[str, Any]:
        """Generate comprehensive rest-of-season projections."""
        try:
            if not projection_models:
                return {"error": "No projection models available"}

            # Get base projection
            ros_projection = await self.get_ros_projection(player, historical_games, upcoming_matchups)

            # Calculate additional metrics
            weekly_projections = []
            for matchup in upcoming_matchups:
                weekly_proj = await self._project_single_game(
                    player, historical_games, matchup, projection_models
                )
                weekly_projections.append(weekly_proj)

            return {
                "total_projection": ros_projection.projected_value,
                "confidence_interval": {
                    "lower": ros_projection.lower_bound,
                    "upper": ros_projection.upper_bound,
                    "confidence_level": ros_projection.confidence_level
                },
                "weekly_projections": weekly_projections,
                "variance": ros_projection.variance,
                "standard_error": ros_projection.standard_error,
                "games_remaining": len(upcoming_matchups)
            }

        except Exception as e:
            logger.error(f"Error generating ROS projections: {str(e)}")
            return {"error": str(e)}

    # Additional helper methods for statistical calculations

    def _stats_to_dataframe(self, stats_list: List[PlayerStats]) -> pd.DataFrame:
        """Convert list of PlayerStats to DataFrame."""
        data = []
        for stat in stats_list:
            stat_dict = stat.dict() if hasattr(stat, 'dict') else stat.__dict__
            data.append(stat_dict)
        return pd.DataFrame(data)

    def _safe_divide(self, numerator: float, denominator: float) -> Optional[float]:
        """Safely divide two numbers, returning None if denominator is 0."""
        if denominator == 0 or denominator is None or numerator is None:
            return None
        return numerator / denominator

    def _calculate_target_share(self, df: pd.DataFrame) -> Optional[float]:
        """Calculate target share (would need team data)."""
        # Placeholder - would need team total targets
        return None

    def _calculate_air_yards_share(self, df: pd.DataFrame) -> Optional[float]:
        """Calculate air yards share (would need air yards data)."""
        # Placeholder - would need air yards data
        return None

    def _calculate_catch_rate(self, df: pd.DataFrame) -> Optional[float]:
        """Calculate catch rate."""
        targets = df['targets'].sum()
        receptions = df['receptions'].sum()
        return self._safe_divide(receptions, targets)

    def _calculate_red_zone_share(self, df: pd.DataFrame) -> Optional[float]:
        """Calculate red zone touch share (would need red zone data)."""
        # Placeholder - would need red zone data
        return None

    def _calculate_goal_line_share(self, df: pd.DataFrame) -> Optional[float]:
        """Calculate goal line touch share (would need goal line data)."""
        # Placeholder - would need goal line data
        return None

    def _calculate_td_efficiency(self, df: pd.DataFrame) -> Optional[float]:
        """Calculate touchdown efficiency."""
        total_tds = df['rushing_touchdowns'].fillna(0).sum() + df['receiving_touchdowns'].fillna(0).sum()
        total_touches = df['rushing_attempts'].fillna(0).sum() + df['receptions'].fillna(0).sum()
        return self._safe_divide(total_tds, total_touches)

    def _calculate_home_away_splits(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate home/away performance splits."""
        # Placeholder - would need home/away indicator
        return {"home_avg": 0.0, "away_avg": 0.0}

    def _calculate_weather_splits(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate weather performance splits."""
        # Placeholder - would need weather data
        return {"good_weather": 0.0, "bad_weather": 0.0}

    def _prepare_modeling_data(
        self,
        player: Player,
        historical_games: List[PlayerStats]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare feature matrix and target for modeling."""
        try:
            df = self._stats_to_dataframe(historical_games)

            # Target variable
            y = df['fantasy_points'].fillna(0).values

            # Feature engineering
            features = []
            feature_names = []

            # Basic volume features
            if player.position in [Position.WR, Position.TE]:
                features.extend([
                    df['targets'].fillna(0),
                    df['receptions'].fillna(0),
                    df['receiving_yards'].fillna(0)
                ])
                feature_names.extend(['targets', 'receptions', 'receiving_yards'])

            if player.position in [Position.RB, Position.QB]:
                features.extend([
                    df['rushing_attempts'].fillna(0),
                    df['rushing_yards'].fillna(0)
                ])
                feature_names.extend(['rushing_attempts', 'rushing_yards'])

            if player.position == Position.QB:
                features.extend([
                    df['passing_attempts'].fillna(0),
                    df['passing_completions'].fillna(0),
                    df['passing_yards'].fillna(0)
                ])
                feature_names.extend(['passing_attempts', 'passing_completions', 'passing_yards'])

            # Time-based features
            features.extend([
                np.arange(len(df)),  # Game number (trend)
                np.arange(len(df)) ** 2  # Quadratic trend
            ])
            feature_names.extend(['game_number', 'game_number_squared'])

            if not features:
                return None, None

            X = np.column_stack(features)

            # Remove any rows with NaN values
            mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            X = X[mask]
            y = y[mask]

            return X, y

        except Exception as e:
            logger.error(f"Error preparing modeling data: {str(e)}")
            return None, None

    async def _fit_linear_models(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, RegressionResults]:
        """Fit various linear regression models."""
        models = {}

        try:
            # Standard linear regression
            lr = LinearRegression()
            lr.fit(X, y)
            lr_pred = lr.predict(X)

            models['linear'] = RegressionResults(
                model_type='linear_regression',
                r2_score=r2_score(y, lr_pred),
                mse=mean_squared_error(y, lr_pred),
                rmse=np.sqrt(mean_squared_error(y, lr_pred)),
                coefficients=dict(zip([f'feature_{i}' for i in range(X.shape[1])], lr.coef_)),
                predictions=lr_pred
            )

            # Ridge regression
            ridge = Ridge(alpha=1.0)
            ridge.fit(X, y)
            ridge_pred = ridge.predict(X)

            models['ridge'] = RegressionResults(
                model_type='ridge_regression',
                r2_score=r2_score(y, ridge_pred),
                mse=mean_squared_error(y, ridge_pred),
                rmse=np.sqrt(mean_squared_error(y, ridge_pred)),
                coefficients=dict(zip([f'feature_{i}' for i in range(X.shape[1])], ridge.coef_)),
                predictions=ridge_pred
            )

            # Elastic Net
            elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)
            elastic.fit(X, y)
            elastic_pred = elastic.predict(X)

            models['elastic_net'] = RegressionResults(
                model_type='elastic_net',
                r2_score=r2_score(y, elastic_pred),
                mse=mean_squared_error(y, elastic_pred),
                rmse=np.sqrt(mean_squared_error(y, elastic_pred)),
                coefficients=dict(zip([f'feature_{i}' for i in range(X.shape[1])], elastic.coef_)),
                predictions=elastic_pred
            )

        except Exception as e:
            logger.error(f"Error fitting linear models: {str(e)}")

        return models

    async def _fit_tree_models(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, RegressionResults]:
        """Fit tree-based models."""
        models = {}

        try:
            # Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            rf_pred = rf.predict(X)

            feature_importance = dict(zip(
                [f'feature_{i}' for i in range(X.shape[1])],
                rf.feature_importances_
            ))

            models['random_forest'] = RegressionResults(
                model_type='random_forest',
                r2_score=r2_score(y, rf_pred),
                mse=mean_squared_error(y, rf_pred),
                rmse=np.sqrt(mean_squared_error(y, rf_pred)),
                feature_importance=feature_importance,
                predictions=rf_pred
            )

            # Gradient Boosting
            gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
            gbr.fit(X, y)
            gbr_pred = gbr.predict(X)

            feature_importance = dict(zip(
                [f'feature_{i}' for i in range(X.shape[1])],
                gbr.feature_importances_
            ))

            models['gradient_boosting'] = RegressionResults(
                model_type='gradient_boosting',
                r2_score=r2_score(y, gbr_pred),
                mse=mean_squared_error(y, gbr_pred),
                rmse=np.sqrt(mean_squared_error(y, gbr_pred)),
                feature_importance=feature_importance,
                predictions=gbr_pred
            )

        except Exception as e:
            logger.error(f"Error fitting tree models: {str(e)}")

        return models

    async def _fit_time_series_model(self, y: np.ndarray) -> Optional[RegressionResults]:
        """Fit ARIMA time series model."""
        if len(y) < 8:  # Need sufficient data for ARIMA
            return None

        try:
            # Find best ARIMA parameters (simplified)
            best_aic = float('inf')
            best_model = None
            best_pred = None

            for p in range(3):
                for d in range(2):
                    for q in range(3):
                        try:
                            model = ARIMA(y, order=(p, d, q))
                            fitted_model = model.fit()
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_model = fitted_model
                                best_pred = fitted_model.fittedvalues
                        except:
                            continue

            if best_model is not None:
                return RegressionResults(
                    model_type='arima',
                    r2_score=r2_score(y, best_pred),
                    mse=mean_squared_error(y, best_pred),
                    rmse=np.sqrt(mean_squared_error(y, best_pred)),
                    predictions=best_pred
                )

        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {str(e)}")

        return None

    def _calculate_matchup_difficulty(self, player: Player, matchup: Matchup) -> float:
        """Calculate matchup difficulty score (0-10 scale)."""
        # Base difficulty
        difficulty = 5.0

        try:
            # Adjust for opponent defense (would need defensive stats)
            # Placeholder logic

            # Weather adjustment
            if matchup.game_environment and matchup.game_environment.is_weather_concern():
                difficulty += 1.0

            # Home/away adjustment
            if player.team == matchup.away_team:  # Away game
                difficulty += 0.5

            return max(0.0, min(10.0, difficulty))

        except Exception as e:
            logger.error(f"Error calculating matchup difficulty: {str(e)}")
            return 5.0

    async def _project_single_game(
        self,
        player: Player,
        historical_games: List[PlayerStats],
        matchup: Matchup,
        projection_models: Dict[str, RegressionResults]
    ) -> float:
        """Project fantasy points for a single game."""
        if not projection_models:
            # Fallback to simple average
            df = self._stats_to_dataframe(historical_games)
            return df['fantasy_points'].fillna(0).mean()

        try:
            # Ensemble prediction from multiple models
            predictions = []

            # Weight models by their R² score
            for model_name, model in projection_models.items():
                if hasattr(model, 'predictions') and model.predictions is not None:
                    # Use the most recent prediction as base
                    base_pred = model.predictions[-1] if len(model.predictions) > 0 else 10.0

                    # Adjust for matchup difficulty
                    difficulty_adj = self._calculate_matchup_difficulty(player, matchup)
                    adjusted_pred = base_pred * (1.0 - (difficulty_adj - 5.0) * 0.1)

                    # Weight by model quality
                    weight = max(0.1, model.r2_score)
                    predictions.append((adjusted_pred, weight))

            if predictions:
                # Weighted average
                total_weight = sum(weight for _, weight in predictions)
                weighted_sum = sum(pred * weight for pred, weight in predictions)
                return max(0.0, weighted_sum / total_weight)
            else:
                # Fallback
                df = self._stats_to_dataframe(historical_games)
                return df['fantasy_points'].fillna(0).mean()

        except Exception as e:
            logger.error(f"Error projecting single game: {str(e)}")
            return 10.0  # Reasonable fallback

    def _calculate_projection_variance(
        self,
        game_projections: List[float],
        projection_models: Dict[str, RegressionResults]
    ) -> float:
        """Calculate variance in projections."""
        if not game_projections:
            return 100.0  # High default variance

        try:
            # Base variance from game projections
            base_variance = np.var(game_projections)

            # Model uncertainty
            model_rmse_values = [
                model.rmse for model in projection_models.values()
                if hasattr(model, 'rmse') and model.rmse is not None
            ]

            if model_rmse_values:
                avg_rmse = np.mean(model_rmse_values)
                model_variance = avg_rmse ** 2
            else:
                model_variance = 25.0  # Default model variance

            # Combine variances
            total_variance = base_variance + model_variance

            return max(1.0, total_variance)  # Minimum variance

        except Exception as e:
            logger.error(f"Error calculating projection variance: {str(e)}")
            return 100.0

    def _calculate_analysis_confidence(
        self,
        data_points: int,
        projection_models: Dict[str, RegressionResults],
        trend_analysis: Optional[TrendAnalysis]
    ) -> float:
        """Calculate overall confidence in analysis (0-1 scale)."""
        try:
            confidence = 0.5  # Base confidence

            # Adjust for data quantity
            if data_points >= 16:
                confidence += 0.3
            elif data_points >= 8:
                confidence += 0.2
            elif data_points >= 4:
                confidence += 0.1

            # Adjust for model quality
            if projection_models:
                avg_r2 = np.mean([
                    model.r2_score for model in projection_models.values()
                    if hasattr(model, 'r2_score') and model.r2_score is not None
                ])
                confidence += avg_r2 * 0.2

            # Adjust for trend clarity
            if trend_analysis and trend_analysis.trend_strength > 0.5:
                confidence += 0.1

            return max(0.0, min(1.0, confidence))

        except Exception as e:
            logger.error(f"Error calculating analysis confidence: {str(e)}")
            return 0.5

    async def _analyze_single_waiver_candidate(
        self,
        player: Player,
        historical_games: List[PlayerStats],
        remaining_schedule: List[Matchup],
        current_roster_pct: float
    ) -> WaiverAnalysis:
        """Analyze a single waiver wire candidate."""
        try:
            # Generate ROS projection
            ros_projection = await self.get_ros_projection(player, historical_games, remaining_schedule)

            # Calculate opportunity score based on recent trends
            df = self._stats_to_dataframe(historical_games)
            recent_avg = df['fantasy_points'].tail(3).mean() if len(df) >= 3 else 0
            season_avg = df['fantasy_points'].mean() if len(df) > 0 else 0

            opportunity_score = max(0, recent_avg - season_avg) * 10  # Boost for improving players

            # Breakout probability (simplified)
            breakout_prob = min(1.0, opportunity_score / 50.0)

            # Generate recommendation
            if ros_projection.projected_value > 100 and current_roster_pct < 20:
                recommendation = "add"
            elif ros_projection.projected_value > 50 and current_roster_pct < 50:
                recommendation = "monitor"
            elif current_roster_pct > 80 and ros_projection.projected_value < 30:
                recommendation = "drop"
            else:
                recommendation = "hold"

            return WaiverAnalysis(
                player_id=player.id,
                current_roster_percentage=current_roster_pct,
                projected_ros_value=ros_projection.projected_value,
                breakout_probability=breakout_prob,
                injury_replacement_value=ros_projection.projected_value * 0.8,  # Conservative
                schedule_strength=5.0,  # Would calculate from matchups
                opportunity_score=opportunity_score,
                recommendation=recommendation,
                confidence=0.7  # Would calculate based on data quality
            )

        except Exception as e:
            logger.error(f"Error analyzing waiver candidate {player.name}: {str(e)}")
            return WaiverAnalysis(
                player_id=player.id,
                current_roster_percentage=current_roster_pct,
                projected_ros_value=0.0,
                breakout_probability=0.0,
                injury_replacement_value=0.0,
                schedule_strength=5.0,
                opportunity_score=0.0,
                recommendation="monitor",
                confidence=0.1
            )

    # Team analysis helper methods

    def _calculate_offensive_efficiency(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate team offensive efficiency metrics."""
        return {
            "points_per_game": df.get('points_per_game', pd.Series([0])).mean(),
            "yards_per_play": df.get('total_yards', pd.Series([0])).sum() / max(1, df.get('total_plays', pd.Series([1])).sum()),
            "red_zone_efficiency": df.get('red_zone_scores', pd.Series([0])).sum() / max(1, df.get('red_zone_trips', pd.Series([1])).sum())
        }

    def _calculate_defensive_efficiency(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate team defensive efficiency metrics."""
        return {
            "points_allowed_per_game": df.get('points_allowed', pd.Series([0])).mean(),
            "yards_allowed_per_play": df.get('yards_allowed', pd.Series([0])).sum() / max(1, df.get('opponent_plays', pd.Series([1])).sum()),
            "turnover_rate": df.get('turnovers_forced', pd.Series([0])).sum() / max(1, len(df))
        }

    def _calculate_pace_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate team pace metrics."""
        return {
            "plays_per_game": df.get('total_plays', pd.Series([0])).mean(),
            "seconds_per_play": df.get('time_of_possession_seconds', pd.Series([0])).sum() / max(1, df.get('total_plays', pd.Series([1])).sum()),
            "neutral_script_pace": 65.0  # Would calculate from situational data
        }

    def _analyze_scoring_trend(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze team scoring trends."""
        points = df.get('points_per_game', pd.Series([0]))
        if len(points) < 2:
            return {"trend": 0.0, "consistency": 0.0}

        # Simple linear trend
        x = np.arange(len(points))
        slope, _, r_value, _, _ = stats.linregress(x, points)

        return {
            "trend": slope,  # Points per game trend
            "consistency": 1.0 - (points.std() / max(0.1, points.mean())),  # Inverse of coefficient of variation
            "recent_avg": points.tail(4).mean(),
            "season_avg": points.mean()
        }

    def _analyze_defensive_trend(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze team defensive trends."""
        points_allowed = df.get('points_allowed', pd.Series([0]))
        if len(points_allowed) < 2:
            return {"trend": 0.0, "consistency": 0.0}

        # Simple linear trend (negative slope is good for defense)
        x = np.arange(len(points_allowed))
        slope, _, r_value, _, _ = stats.linregress(x, points_allowed)

        return {
            "trend": -slope,  # Flip sign so positive is good
            "consistency": 1.0 - (points_allowed.std() / max(0.1, points_allowed.mean())),
            "recent_avg": points_allowed.tail(4).mean(),
            "season_avg": points_allowed.mean()
        }

    async def _analyze_strength_of_schedule(
        self,
        team: Team,
        upcoming_matchups: List[Matchup]
    ) -> Dict[str, Any]:
        """Analyze strength of schedule for upcoming matchups."""
        if not upcoming_matchups:
            return {"sos_rating": 5.0, "games": 0}

        try:
            # Placeholder SOS calculation
            # In reality, would use opponent defensive rankings, etc.
            sos_scores = []

            for matchup in upcoming_matchups:
                # Base score
                score = 5.0

                # Adjust for home/away
                if team == matchup.away_team:
                    score += 0.5  # Road games are harder

                # Weather adjustment
                if matchup.game_environment and matchup.game_environment.is_weather_concern():
                    score += 1.0

                sos_scores.append(score)

            return {
                "sos_rating": statistics.mean(sos_scores),
                "games": len(upcoming_matchups),
                "home_games": sum(1 for m in upcoming_matchups if team == m.home_team),
                "away_games": sum(1 for m in upcoming_matchups if team == m.away_team),
                "weather_games": sum(1 for m in upcoming_matchups
                                   if m.game_environment and m.game_environment.is_weather_concern())
            }

        except Exception as e:
            logger.error(f"Error analyzing strength of schedule: {str(e)}")
            return {"sos_rating": 5.0, "games": len(upcoming_matchups)}

    def __del__(self):
        """Cleanup executors on deletion."""
        try:
            if hasattr(self, 'thread_executor'):
                self.thread_executor.shutdown(wait=False)
            if hasattr(self, 'process_executor'):
                self.process_executor.shutdown(wait=False)
        except:
            pass
