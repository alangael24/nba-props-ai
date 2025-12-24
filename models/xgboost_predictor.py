"""
Fase 2: Modelo Predictivo con XGBoost
NUEVO ENFOQUE: Separación de Variables (Minutos × Eficiencia)

P = T × E (Puntos = Tiempo × Eficiencia)

Modelo A (Minutos): Depende del entrenador, faltas, blowouts
Modelo B (PPM/RPM/APM): Depende de habilidad y defensa rival
"""

import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from scipy import stats
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb

DB_PATH = Path(__file__).parent.parent / "data" / "nba_props.db"
MODELS_PATH = Path(__file__).parent


class OpponentStats:
    """
    Carga y cachea estadísticas de oponentes para usar como features.

    Incluye:
    - DvP (Defense vs Position): Cuánto permite cada equipo a cada posición
    - Pace proxy: Ritmo de juego aproximado del oponente
    - Defensive rating proxy: Eficiencia defensiva del oponente
    """

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DB_PATH
        self.dvp_cache: dict = {}
        self.team_stats_cache: dict = {}
        self._load_data()

    def _load_data(self):
        """Carga DvP y calcula stats de equipos."""
        try:
            conn = sqlite3.connect(self.db_path)

            # Cargar DvP
            dvp_df = pd.read_sql("""
                SELECT team_abbrev, position, pts_allowed_avg, reb_allowed_avg,
                       ast_allowed_avg, games_sample
                FROM defense_vs_position
                WHERE season = '2024-25'
            """, conn)

            for _, row in dvp_df.iterrows():
                key = (row["team_abbrev"], row["position"])
                self.dvp_cache[key] = {
                    "pts_allowed": row["pts_allowed_avg"],
                    "reb_allowed": row["reb_allowed_avg"],
                    "ast_allowed": row["ast_allowed_avg"],
                    "games": row["games_sample"]
                }

            # Calcular stats de equipos (pace proxy, def rating proxy)
            team_stats = pd.read_sql("""
                SELECT
                    opponent_abbrev as team,
                    AVG(pts) as opp_pts_allowed,
                    COUNT(*) as games,
                    AVG(min) as avg_game_min
                FROM player_game_logs
                WHERE opponent_abbrev IS NOT NULL
                AND min > 0
                GROUP BY opponent_abbrev
                HAVING games >= 50
            """, conn)

            # También calculamos cuántos puntos anota el oponente (su ofensiva)
            team_offense = pd.read_sql("""
                SELECT
                    t.abbreviation as team,
                    AVG(g.pts) as team_pts_scored
                FROM player_game_logs g
                JOIN players p ON g.player_id = p.player_id
                JOIN teams t ON p.team_id = t.team_id
                WHERE g.min > 0
                GROUP BY t.abbreviation
            """, conn)

            conn.close()

            # Combinar y calcular proxies
            if not team_stats.empty:
                # Pace proxy: más puntos permitidos = más ritmo
                mean_pts = team_stats["opp_pts_allowed"].mean()
                std_pts = team_stats["opp_pts_allowed"].std()

                for _, row in team_stats.iterrows():
                    team = row["team"]
                    pts_allowed = row["opp_pts_allowed"]

                    # Pace proxy normalizado (z-score, luego a escala 0.8-1.2)
                    pace_z = (pts_allowed - mean_pts) / std_pts if std_pts > 0 else 0
                    pace_factor = 1.0 + (pace_z * 0.1)  # ±10% around baseline
                    pace_factor = max(0.8, min(1.2, pace_factor))

                    # Def rating proxy: más puntos permitidos = peor defensa
                    def_rating = pts_allowed / mean_pts if mean_pts > 0 else 1.0

                    self.team_stats_cache[team] = {
                        "pace_factor": pace_factor,
                        "def_rating": def_rating,
                        "pts_allowed_avg": pts_allowed
                    }

            print(f"OpponentStats: Cargados {len(self.dvp_cache)} DvP entries, {len(self.team_stats_cache)} team stats")

        except Exception as e:
            print(f"Warning: Error cargando opponent stats: {e}")

    def get_dvp(self, team_abbrev: str, position: str) -> dict:
        """Obtiene DvP para equipo/posición."""
        key = (team_abbrev, position)
        if key in self.dvp_cache:
            return self.dvp_cache[key]

        # Fallback: promedio de todas las posiciones del equipo
        team_dvps = [v for k, v in self.dvp_cache.items() if k[0] == team_abbrev]
        if team_dvps:
            return {
                "pts_allowed": np.mean([d["pts_allowed"] for d in team_dvps]),
                "reb_allowed": np.mean([d["reb_allowed"] for d in team_dvps]),
                "ast_allowed": np.mean([d["ast_allowed"] for d in team_dvps]),
                "games": 0
            }

        # Default neutral
        return {"pts_allowed": 15.0, "reb_allowed": 5.0, "ast_allowed": 3.0, "games": 0}

    def get_team_stats(self, team_abbrev: str) -> dict:
        """Obtiene pace y def_rating para un equipo."""
        if team_abbrev in self.team_stats_cache:
            return self.team_stats_cache[team_abbrev]

        # Default neutral
        return {"pace_factor": 1.0, "def_rating": 1.0, "pts_allowed_avg": 100.0}

    def infer_position(self, player_df: pd.DataFrame) -> str:
        """Infiere posición del jugador basado en sus stats."""
        if player_df.empty or player_df["min"].mean() < 1:
            return "SF"  # Default

        avg_ast = player_df["ast"].mean()
        avg_reb = player_df["reb"].mean()
        avg_min = player_df["min"].mean()

        apm = avg_ast / avg_min if avg_min > 0 else 0
        rpm = avg_reb / avg_min if avg_min > 0 else 0

        if apm > 0.25:
            return "PG"
        elif rpm > 0.30 and apm < 0.10:
            return "C"
        elif rpm > 0.25:
            return "PF"
        elif apm > 0.15:
            return "SG"
        else:
            return "SF"


# Singleton para reusar
_opponent_stats: OpponentStats = None

def get_opponent_stats() -> OpponentStats:
    """Obtiene instancia singleton de OpponentStats."""
    global _opponent_stats
    if _opponent_stats is None:
        _opponent_stats = OpponentStats()
    return _opponent_stats


class ProbabilityCalibrator:
    """
    Calibra probabilidades usando Isotonic Regression.

    Problema: Si el modelo dice "60% prob de Over", pero históricamente
    cuando dice 60% solo acierta el 52%, tenemos un sesgo.

    Solución: Usar regresión isotónica para mapear probabilidades
    crudas a probabilidades calibradas basadas en datos históricos.
    """

    def __init__(self):
        self.iso_reg = IsotonicRegression(out_of_bounds='clip')
        self.is_fitted = False

    def fit(self, predicted_probs: np.ndarray, actual_outcomes: np.ndarray):
        """
        Ajusta el calibrador con datos históricos.

        Args:
            predicted_probs: Probabilidades crudas del modelo (0-1)
            actual_outcomes: 1 si fue Over, 0 si fue Under
        """
        # Necesitamos suficientes datos para calibrar
        if len(predicted_probs) < 100:
            print("  ⚠️ Pocos datos para calibración isotónica (<100)")
            return

        self.iso_reg.fit(predicted_probs, actual_outcomes)
        self.is_fitted = True

        # Mostrar estadísticas de calibración
        calibrated = self.iso_reg.predict(predicted_probs)
        print(f"  Calibración isotónica ajustada:")
        print(f"    - Datos: {len(predicted_probs)}")
        print(f"    - Prob cruda rango: [{predicted_probs.min():.2f}, {predicted_probs.max():.2f}]")
        print(f"    - Prob calibrada rango: [{calibrated.min():.2f}, {calibrated.max():.2f}]")

    def calibrate(self, raw_prob: float) -> float:
        """
        Calibra una probabilidad cruda.

        Args:
            raw_prob: Probabilidad del modelo (0-1)

        Returns:
            Probabilidad calibrada
        """
        if not self.is_fitted:
            return raw_prob

        return float(self.iso_reg.predict([raw_prob])[0])

    def calibrate_batch(self, raw_probs: np.ndarray) -> np.ndarray:
        """Calibra un array de probabilidades."""
        if not self.is_fitted:
            return raw_probs
        return self.iso_reg.predict(raw_probs)


class NBAPropsPredictor:
    """
    Modelo de predicción para Player Props de la NBA.

    ARQUITECTURA:
    1. Separación de Variables: Minutos × Eficiencia
    2. Regresión de Cuantiles: P15 (piso), P50 (mediana), P85 (techo)

    Ventaja: No asumimos distribución normal. Capturamos la forma real
    de la incertidumbre de cada jugador.
    """

    def __init__(self):
        # Modelos de MINUTOS (con cuantiles)
        self.model_minutes_p15 = None  # Piso de minutos
        self.model_minutes_p50 = None  # Mediana de minutos
        self.model_minutes_p85 = None  # Techo de minutos

        # Modelos de EFICIENCIA PPM (con cuantiles)
        self.model_ppm_p15 = None
        self.model_ppm_p50 = None
        self.model_ppm_p85 = None

        # Modelos RPM (con cuantiles)
        self.model_rpm_p15 = None
        self.model_rpm_p50 = None
        self.model_rpm_p85 = None

        # Modelos APM (con cuantiles)
        self.model_apm_p15 = None
        self.model_apm_p50 = None
        self.model_apm_p85 = None

        self.minutes_features = None
        self.efficiency_features = None

        # Calibración de probabilidades
        self.calibration_curve = None
        self.probability_calibrator = ProbabilityCalibrator()

        # Cuantiles que usamos
        self.quantiles = [0.15, 0.50, 0.85]

    def create_features(self, df: pd.DataFrame, include_opponent_stats: bool = True) -> pd.DataFrame:
        """
        Crea features separadas para Minutos y Eficiencia.

        Args:
            df: DataFrame con game logs
            include_opponent_stats: Si True, incluye DvP y pace del oponente
        """
        df = df.copy()
        df = df.sort_values(["player_id", "game_date"])
        df["game_date"] = pd.to_datetime(df["game_date"])

        # Calcular eficiencias (por minuto)
        df["ppm"] = df["pts"] / df["min"].replace(0, 1)  # Points per minute
        df["rpm"] = df["reb"] / df["min"].replace(0, 1)  # Rebounds per minute
        df["apm"] = df["ast"] / df["min"].replace(0, 1)  # Assists per minute

        # Cargar opponent stats si está habilitado
        opp_stats = get_opponent_stats() if include_opponent_stats else None

        features = []

        for player_id in df["player_id"].unique():
            player_df = df[df["player_id"] == player_id].copy()

            if len(player_df) < 10:
                continue

            # Inferir posición del jugador para DvP lookup
            player_position = opp_stats.infer_position(player_df) if opp_stats else "SF"

            for i in range(10, len(player_df)):
                row = player_df.iloc[i]
                historical = player_df.iloc[:i]

                last_5 = historical.tail(5)
                last_10 = historical.tail(10)

                # === FEATURES PARA MODELO DE MINUTOS ===
                # (Dependen del contexto: entrenador, rotación, blowouts)

                avg_min_5 = last_5["min"].mean()
                avg_min_10 = last_10["min"].mean()
                season_avg_min = historical["min"].mean()
                std_min = historical["min"].std()
                min_trend = avg_min_5 - avg_min_10

                # Consistencia de minutos (¿es titular estable?)
                min_consistency = 1 - (std_min / season_avg_min) if season_avg_min > 0 else 0

                # Días de descanso (afecta minutos en B2B)
                if i > 0:
                    prev_game = player_df.iloc[i - 1]
                    days_rest = (row["game_date"] - prev_game["game_date"]).days
                else:
                    days_rest = 3
                days_rest = min(days_rest, 7)
                is_b2b = 1 if days_rest <= 1 else 0

                # Local/Visitante
                is_home = row["is_home"]

                # Minutos en partidos similares (home/away)
                similar_games = historical[historical["is_home"] == is_home]
                avg_min_similar = similar_games["min"].mean() if len(similar_games) > 0 else season_avg_min

                # === FEATURES PARA MODELO DE EFICIENCIA ===
                # (Dependen de habilidad vs defensa rival)

                opponent = row["opponent_abbrev"]
                vs_opponent = historical[historical["opponent_abbrev"] == opponent]

                # Eficiencia histórica
                avg_ppm_5 = last_5["ppm"].mean()
                avg_ppm_10 = last_10["ppm"].mean()
                season_avg_ppm = historical["ppm"].mean()
                std_ppm = historical["ppm"].std()

                avg_rpm_5 = last_5["rpm"].mean()
                avg_rpm_10 = last_10["rpm"].mean()
                season_avg_rpm = historical["rpm"].mean()
                std_rpm = historical["rpm"].std()

                avg_apm_5 = last_5["apm"].mean()
                avg_apm_10 = last_10["apm"].mean()
                season_avg_apm = historical["apm"].mean()
                std_apm = historical["apm"].std()

                # Eficiencia vs oponente específico
                if len(vs_opponent) >= 2:
                    vs_opp_ppm = vs_opponent["ppm"].mean()
                    vs_opp_rpm = vs_opponent["rpm"].mean()
                    vs_opp_apm = vs_opponent["apm"].mean()
                else:
                    vs_opp_ppm = season_avg_ppm
                    vs_opp_rpm = season_avg_rpm
                    vs_opp_apm = season_avg_apm

                # Tendencia de eficiencia
                ppm_trend = avg_ppm_5 - avg_ppm_10
                rpm_trend = avg_rpm_5 - avg_rpm_10
                apm_trend = avg_apm_5 - avg_apm_10

                # === OPPONENT STATS (DvP, Pace, Def Rating) ===
                if opp_stats:
                    dvp = opp_stats.get_dvp(opponent, player_position)
                    team_stats = opp_stats.get_team_stats(opponent)

                    opp_dvp_pts = dvp["pts_allowed"]
                    opp_dvp_reb = dvp["reb_allowed"]
                    opp_dvp_ast = dvp["ast_allowed"]
                    opp_pace = team_stats["pace_factor"]
                    opp_def_rating = team_stats["def_rating"]
                else:
                    # Defaults neutrales
                    opp_dvp_pts = 15.0
                    opp_dvp_reb = 5.0
                    opp_dvp_ast = 3.0
                    opp_pace = 1.0
                    opp_def_rating = 1.0

                # === CREAR FILA ===
                feature_row = {
                    # IDs
                    "player_id": row["player_id"],
                    "player_name": row["player_name"],
                    "game_date": row["game_date"],
                    "game_id": row["game_id"],
                    "opponent": opponent,

                    # FEATURES MINUTOS
                    "avg_min_5": avg_min_5,
                    "avg_min_10": avg_min_10,
                    "season_avg_min": season_avg_min,
                    "std_min": std_min,
                    "min_trend": min_trend,
                    "min_consistency": min_consistency,
                    "days_rest": days_rest,
                    "is_b2b": is_b2b,
                    "is_home": is_home,
                    "avg_min_similar": avg_min_similar,

                    # FEATURES EFICIENCIA
                    "avg_ppm_5": avg_ppm_5,
                    "avg_ppm_10": avg_ppm_10,
                    "season_avg_ppm": season_avg_ppm,
                    "std_ppm": std_ppm,
                    "ppm_trend": ppm_trend,
                    "vs_opp_ppm": vs_opp_ppm,

                    "avg_rpm_5": avg_rpm_5,
                    "avg_rpm_10": avg_rpm_10,
                    "season_avg_rpm": season_avg_rpm,
                    "std_rpm": std_rpm,
                    "rpm_trend": rpm_trend,
                    "vs_opp_rpm": vs_opp_rpm,

                    "avg_apm_5": avg_apm_5,
                    "avg_apm_10": avg_apm_10,
                    "season_avg_apm": season_avg_apm,
                    "std_apm": std_apm,
                    "apm_trend": apm_trend,
                    "vs_opp_apm": vs_opp_apm,

                    # OPPONENT STATS (DvP + Pace + DefRating)
                    "opp_dvp_pts": opp_dvp_pts,
                    "opp_dvp_reb": opp_dvp_reb,
                    "opp_dvp_ast": opp_dvp_ast,
                    "opp_pace": opp_pace,
                    "opp_def_rating": opp_def_rating,

                    # TARGETS
                    "actual_min": row["min"],
                    "actual_pts": row["pts"],
                    "actual_reb": row["reb"],
                    "actual_ast": row["ast"],
                    "actual_ppm": row["ppm"],
                    "actual_rpm": row["rpm"],
                    "actual_apm": row["apm"],
                }

                features.append(feature_row)

        return pd.DataFrame(features)

    def get_minutes_features(self):
        """Features para predecir MINUTOS."""
        return [
            "avg_min_5", "avg_min_10", "season_avg_min", "std_min",
            "min_trend", "min_consistency", "days_rest", "is_b2b",
            "is_home", "avg_min_similar"
        ]

    def get_efficiency_features(self):
        """Features para predecir EFICIENCIA (incluye opponent stats)."""
        return [
            # PPM features
            "avg_ppm_5", "avg_ppm_10", "season_avg_ppm", "std_ppm",
            "ppm_trend", "vs_opp_ppm",
            # RPM features
            "avg_rpm_5", "avg_rpm_10", "season_avg_rpm", "std_rpm",
            "rpm_trend", "vs_opp_rpm",
            # APM features
            "avg_apm_5", "avg_apm_10", "season_avg_apm", "std_apm",
            "apm_trend", "vs_opp_apm",
            # Context
            "is_home",
            # Opponent stats (DvP + Pace + DefRating)
            "opp_dvp_pts", "opp_dvp_reb", "opp_dvp_ast",
            "opp_pace", "opp_def_rating"
        ]

    def _train_quantile_model(self, X_train, y_train, quantile: float,
                               n_estimators=150, max_depth=5):
        """Entrena un modelo de regresión de cuantiles."""
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.05,
            subsample=0.8,
            objective='reg:quantileerror',
            quantile_alpha=quantile,
            random_state=42
        )
        model.fit(X_train, y_train)
        return model

    def _calibrate_probabilities(self, actual: np.ndarray, pred_p15: np.ndarray,
                                  pred_p50: np.ndarray, pred_p85: np.ndarray) -> dict:
        """
        Calibra probabilidades empíricas basadas en datos históricos.

        Para cada posición del ACTUAL relativo a los cuantiles predichos,
        calcula el % real de Over observado en diferentes líneas hipotéticas.
        """
        n = len(actual)

        # Calibración 1: ¿Qué % de veces el actual supera cada cuantil?
        actual_above_p15 = (actual > pred_p15).mean()
        actual_above_p50 = (actual > pred_p50).mean()
        actual_above_p85 = (actual > pred_p85).mean()

        print(f"Calibración de cuantiles:")
        print(f"  Actual > P15: {actual_above_p15*100:.1f}% (esperado ~85%)")
        print(f"  Actual > P50: {actual_above_p50*100:.1f}% (esperado ~50%)")
        print(f"  Actual > P85: {actual_above_p85*100:.1f}% (esperado ~15%)")

        # Calibración 2: Para diferentes "líneas" relativas a cuantiles,
        # ¿cuál es la tasa de Over real?
        # Simulamos líneas en diferentes posiciones relativas a los cuantiles
        calibration_points = []

        for pct in [0.0, 0.25, 0.5, 0.75, 1.0]:
            # Línea interpolada entre P15 y P85
            if pct <= 0.5:
                # Entre P15 y P50
                hypothetical_line = pred_p15 + (pred_p50 - pred_p15) * (pct / 0.5)
            else:
                # Entre P50 y P85
                hypothetical_line = pred_p50 + (pred_p85 - pred_p50) * ((pct - 0.5) / 0.5)

            over_rate = (actual > hypothetical_line).mean()
            calibration_points.append({
                "quantile_position": pct,  # 0=P15, 0.5=P50, 1.0=P85
                "empirical_over_rate": over_rate,
                "expected_over_rate": 0.85 - pct * 0.70  # 85% at P15, 15% at P85
            })

        print(f"\nCalibración por posición de línea:")
        for point in calibration_points:
            pos = point["quantile_position"]
            emp = point["empirical_over_rate"] * 100
            exp = point["expected_over_rate"] * 100
            label = "P15" if pos == 0 else "P50" if pos == 0.5 else "P85" if pos == 1.0 else f"{pos:.0%}"
            print(f"  Línea @ {label}: {emp:.1f}% over (esperado {exp:.1f}%)")

        # Calibración 3: Bins de error relativo para isotonic regression
        diffs = (actual - pred_p50) / np.maximum(pred_p50, 1)
        bins = np.percentile(diffs, np.arange(0, 101, 10))

        detailed_calibration = []
        for i in range(len(bins) - 1):
            mask = (diffs >= bins[i]) & (diffs < bins[i + 1])
            if mask.sum() > 0:
                over_rate = (actual[mask] > pred_p50[mask]).mean()
                detailed_calibration.append({
                    "diff_pct_low": float(bins[i]),
                    "diff_pct_high": float(bins[i + 1]),
                    "empirical_over_rate": float(over_rate),
                    "count": int(mask.sum())
                })

        return {
            "quantile_calibration": {
                "actual_above_p15": float(actual_above_p15),
                "actual_above_p50": float(actual_above_p50),
                "actual_above_p85": float(actual_above_p85),
            },
            "line_position_calibration": calibration_points,
            "detailed": detailed_calibration
        }

    def train(self, test_size: float = 0.2):
        """
        Entrena modelos con REGRESIÓN DE CUANTILES.

        Para cada variable predecimos:
        - P15: El piso (mal día)
        - P50: La mediana (día normal)
        - P85: El techo (buen día)

        IMPORTANTE: Split TEMPORAL por fecha, no por índice.
        Después de evaluar, re-entrena con TODO el dataset.
        """
        print("="*60)
        print("ENTRENAMIENTO: Minutos × Eficiencia + CUANTILES")
        print("="*60)
        print("Cuantiles: P15 (piso), P50 (mediana), P85 (techo)")

        print("\nCargando datos...")
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("""
            SELECT * FROM player_game_logs
            WHERE min > 10 AND player_name IS NOT NULL
            ORDER BY game_date
        """, conn)
        conn.close()

        print(f"Registros cargados: {len(df)}")

        print("Creando features...")
        features_df = self.create_features(df)
        print(f"Registros con features: {len(features_df)}")

        if features_df.empty:
            print("ERROR: No hay suficientes datos.")
            return

        # === SPLIT TEMPORAL POR FECHA (no por índice) ===
        features_df = features_df.sort_values("game_date").reset_index(drop=True)
        cutoff_date = features_df["game_date"].quantile(1 - test_size)
        train_mask = features_df["game_date"] < cutoff_date
        test_mask = features_df["game_date"] >= cutoff_date

        print(f"Split temporal: train < {cutoff_date.strftime('%Y-%m-%d')}, test >= {cutoff_date.strftime('%Y-%m-%d')}")
        print(f"Train: {train_mask.sum()} | Test: {test_mask.sum()}")

        self.minutes_features = self.get_minutes_features()
        self.efficiency_features = self.get_efficiency_features()

        # ============================================================
        # MODELO A: MINUTOS (con cuantiles)
        # ============================================================
        print("\n" + "="*50)
        print("MODELO A: PREDICCIÓN DE MINUTOS (Cuantiles)")
        print("="*50)

        X_min = features_df[self.minutes_features].fillna(0)
        y_min = features_df["actual_min"]

        X_min_train = X_min[train_mask]
        X_min_test = X_min[test_mask]
        y_min_train = y_min[train_mask]
        y_min_test = y_min[test_mask]

        # Entrenar cuantiles para minutos
        print("Entrenando P15 (piso)...")
        self.model_minutes_p15 = self._train_quantile_model(X_min_train, y_min_train, 0.15)

        print("Entrenando P50 (mediana)...")
        self.model_minutes_p50 = self._train_quantile_model(X_min_train, y_min_train, 0.50)

        print("Entrenando P85 (techo)...")
        self.model_minutes_p85 = self._train_quantile_model(X_min_train, y_min_train, 0.85)

        # Evaluar
        pred_min_p15 = self.model_minutes_p15.predict(X_min_test)
        pred_min_p50 = self.model_minutes_p50.predict(X_min_test)
        pred_min_p85 = self.model_minutes_p85.predict(X_min_test)

        mae_min = mean_absolute_error(y_min_test, pred_min_p50)
        print(f"\nMAE Minutos (mediana): {mae_min:.2f}")

        # Verificar cobertura de cuantiles
        coverage_15 = (y_min_test.values > pred_min_p15).mean()
        coverage_85 = (y_min_test.values < pred_min_p85).mean()
        print(f"Cobertura P15: {coverage_15*100:.1f}% (esperado ~85%)")
        print(f"Cobertura P85: {coverage_85*100:.1f}% (esperado ~85%)")

        # ============================================================
        # MODELO B: EFICIENCIA PPM (con cuantiles)
        # ============================================================
        print("\n" + "="*50)
        print("MODELO B: EFICIENCIA PPM (Cuantiles)")
        print("="*50)

        ppm_features = [
            "avg_ppm_5", "avg_ppm_10", "season_avg_ppm", "std_ppm",
            "ppm_trend", "vs_opp_ppm", "is_home"
        ]

        X_ppm = features_df[ppm_features].fillna(0)
        y_ppm = features_df["actual_ppm"]

        X_ppm_train = X_ppm[train_mask]
        y_ppm_train = y_ppm[train_mask]
        X_ppm_test = X_ppm[test_mask]
        y_ppm_test = y_ppm[test_mask]

        print("Entrenando PPM cuantiles...")
        self.model_ppm_p15 = self._train_quantile_model(X_ppm_train, y_ppm_train, 0.15, max_depth=4)
        self.model_ppm_p50 = self._train_quantile_model(X_ppm_train, y_ppm_train, 0.50, max_depth=4)
        self.model_ppm_p85 = self._train_quantile_model(X_ppm_train, y_ppm_train, 0.85, max_depth=4)

        pred_ppm_p50 = self.model_ppm_p50.predict(X_ppm_test)
        mae_ppm = mean_absolute_error(y_ppm_test, pred_ppm_p50)
        print(f"MAE PPM (mediana): {mae_ppm:.3f}")

        # ============================================================
        # MODELOS RPM y APM (con cuantiles P15/P50/P85)
        # ============================================================
        print("\n" + "="*50)
        print("MODELOS RPM y APM (Cuantiles)")
        print("="*50)

        rpm_features = [
            "avg_rpm_5", "avg_rpm_10", "season_avg_rpm", "std_rpm",
            "rpm_trend", "vs_opp_rpm", "is_home"
        ]
        X_rpm = features_df[rpm_features].fillna(0)
        y_rpm = features_df["actual_rpm"]

        X_rpm_train = X_rpm[train_mask]
        X_rpm_test = X_rpm[test_mask]
        y_rpm_train = y_rpm[train_mask]
        y_rpm_test = y_rpm[test_mask]

        print("Entrenando RPM cuantiles...")
        self.model_rpm_p15 = self._train_quantile_model(X_rpm_train, y_rpm_train, 0.15, max_depth=4)
        self.model_rpm_p50 = self._train_quantile_model(X_rpm_train, y_rpm_train, 0.50, max_depth=4)
        self.model_rpm_p85 = self._train_quantile_model(X_rpm_train, y_rpm_train, 0.85, max_depth=4)

        pred_rpm_p50 = self.model_rpm_p50.predict(X_rpm_test)
        mae_rpm = mean_absolute_error(y_rpm_test, pred_rpm_p50)
        print(f"MAE RPM (mediana): {mae_rpm:.3f}")

        apm_features = [
            "avg_apm_5", "avg_apm_10", "season_avg_apm", "std_apm",
            "apm_trend", "vs_opp_apm", "is_home"
        ]
        X_apm = features_df[apm_features].fillna(0)
        y_apm = features_df["actual_apm"]

        X_apm_train = X_apm[train_mask]
        X_apm_test = X_apm[test_mask]
        y_apm_train = y_apm[train_mask]
        y_apm_test = y_apm[test_mask]

        print("Entrenando APM cuantiles...")
        self.model_apm_p15 = self._train_quantile_model(X_apm_train, y_apm_train, 0.15, max_depth=4)
        self.model_apm_p50 = self._train_quantile_model(X_apm_train, y_apm_train, 0.50, max_depth=4)
        self.model_apm_p85 = self._train_quantile_model(X_apm_train, y_apm_train, 0.85, max_depth=4)

        pred_apm_p50 = self.model_apm_p50.predict(X_apm_test)
        mae_apm = mean_absolute_error(y_apm_test, pred_apm_p50)
        print(f"MAE APM (mediana): {mae_apm:.3f}")

        # ============================================================
        # VALIDACIÓN: Predicción Combinada con Cuantiles
        # ============================================================
        print("\n" + "="*50)
        print("VALIDACIÓN: Puntos = Minutos × PPM (con cuantiles)")
        print("="*50)

        # Predicción mediana
        final_pred_pts_p50 = pred_min_p50 * pred_ppm_p50

        # Predicción piso (min bajo × ppm bajo)
        pred_ppm_p15 = self.model_ppm_p15.predict(X_ppm_test)
        final_pred_pts_p15 = pred_min_p15 * pred_ppm_p15

        # Predicción techo (min alto × ppm alto)
        pred_ppm_p85 = self.model_ppm_p85.predict(X_ppm_test)
        final_pred_pts_p85 = pred_min_p85 * pred_ppm_p85

        actual_pts = features_df["actual_pts"][test_mask]

        mae_pts = mean_absolute_error(actual_pts, final_pred_pts_p50)
        print(f"MAE Puntos (mediana): {mae_pts:.2f}")

        # Verificar cobertura de cuantiles en puntos finales
        pts_above_p15 = (actual_pts.values > final_pred_pts_p15).mean()
        pts_below_p85 = (actual_pts.values < final_pred_pts_p85).mean()
        print(f"Puntos > P15: {pts_above_p15*100:.1f}% (esperado ~85%)")
        print(f"Puntos < P85: {pts_below_p85*100:.1f}% (esperado ~85%)")

        # Validación REB/AST con cuantiles
        pred_rpm_p15 = self.model_rpm_p15.predict(X_rpm_test)
        pred_rpm_p85 = self.model_rpm_p85.predict(X_rpm_test)
        pred_apm_p15 = self.model_apm_p15.predict(X_apm_test)
        pred_apm_p85 = self.model_apm_p85.predict(X_apm_test)

        final_pred_reb_p15 = pred_min_p15 * pred_rpm_p15
        final_pred_reb_p50 = pred_min_p50 * pred_rpm_p50
        final_pred_reb_p85 = pred_min_p85 * pred_rpm_p85

        final_pred_ast_p15 = pred_min_p15 * pred_apm_p15
        final_pred_ast_p50 = pred_min_p50 * pred_apm_p50
        final_pred_ast_p85 = pred_min_p85 * pred_apm_p85

        actual_reb = features_df["actual_reb"][test_mask]
        actual_ast = features_df["actual_ast"][test_mask]

        mae_reb = mean_absolute_error(actual_reb, final_pred_reb_p50)
        mae_ast = mean_absolute_error(actual_ast, final_pred_ast_p50)
        print(f"MAE Rebotes (mediana): {mae_reb:.2f}")
        print(f"MAE Asistencias (mediana): {mae_ast:.2f}")

        # Ejemplo de rango
        print(f"\nEjemplo de predicción con rango (promedio test set):")
        print(f"  PTS: {final_pred_pts_p15.mean():.1f} - {final_pred_pts_p50.mean():.1f} - {final_pred_pts_p85.mean():.1f}")
        print(f"  REB: {final_pred_reb_p15.mean():.1f} - {final_pred_reb_p50.mean():.1f} - {final_pred_reb_p85.mean():.1f}")
        print(f"  AST: {final_pred_ast_p15.mean():.1f} - {final_pred_ast_p50.mean():.1f} - {final_pred_ast_p85.mean():.1f}")

        # ============================================================
        # CALIBRACIÓN DE PROBABILIDADES (guardar para uso posterior)
        # ============================================================
        print("\n" + "="*50)
        print("CALIBRACIÓN DE PROBABILIDADES")
        print("="*50)

        # Calcular calibración empírica: para cada rango de predicción,
        # ¿qué porcentaje realmente supera la línea?
        calibration_data = self._calibrate_probabilities(
            actual_pts.values, final_pred_pts_p15, final_pred_pts_p50, final_pred_pts_p85
        )
        self.calibration_curve = calibration_data

        # Calibración isotónica para ajustar probabilidades
        print("\nAjustando calibración isotónica...")

        # Generar probabilidades crudas para cada predicción del test set
        # Usamos múltiples líneas hipotéticas para tener más datos de calibración
        raw_probs = []
        actual_outcomes = []

        for i in range(len(actual_pts)):
            # Probamos varias líneas alrededor de P50 para cada predicción
            for line_offset in [-3, -1.5, 0, 1.5, 3]:
                hypothetical_line = final_pred_pts_p50[i] + line_offset
                if hypothetical_line <= 0:
                    continue

                # Probabilidad cruda de over
                raw_prob = self.calculate_probability_from_quantiles(
                    final_pred_pts_p15[i],
                    final_pred_pts_p50[i],
                    final_pred_pts_p85[i],
                    hypothetical_line
                )
                raw_probs.append(raw_prob)

                # Outcome real: ¿el actual superó la línea hipotética?
                actual_over = 1 if actual_pts.values[i] > hypothetical_line else 0
                actual_outcomes.append(actual_over)

        raw_probs = np.array(raw_probs)
        actual_outcomes = np.array(actual_outcomes)

        self.probability_calibrator.fit(raw_probs, actual_outcomes)

        # ============================================================
        # RE-ENTRENAR CON TODOS LOS DATOS PARA PRODUCCIÓN
        # ============================================================
        print("\n" + "="*50)
        print("RE-ENTRENANDO CON TODOS LOS DATOS PARA PRODUCCIÓN")
        print("="*50)

        # Minutos - todo el dataset
        print("Re-entrenando modelos de minutos...")
        self.model_minutes_p15 = self._train_quantile_model(X_min, y_min, 0.15)
        self.model_minutes_p50 = self._train_quantile_model(X_min, y_min, 0.50)
        self.model_minutes_p85 = self._train_quantile_model(X_min, y_min, 0.85)

        # PPM - todo el dataset
        print("Re-entrenando modelos de PPM...")
        self.model_ppm_p15 = self._train_quantile_model(X_ppm, y_ppm, 0.15, max_depth=4)
        self.model_ppm_p50 = self._train_quantile_model(X_ppm, y_ppm, 0.50, max_depth=4)
        self.model_ppm_p85 = self._train_quantile_model(X_ppm, y_ppm, 0.85, max_depth=4)

        # RPM - todo el dataset
        print("Re-entrenando modelos de RPM...")
        self.model_rpm_p15 = self._train_quantile_model(X_rpm, y_rpm, 0.15, max_depth=4)
        self.model_rpm_p50 = self._train_quantile_model(X_rpm, y_rpm, 0.50, max_depth=4)
        self.model_rpm_p85 = self._train_quantile_model(X_rpm, y_rpm, 0.85, max_depth=4)

        # APM - todo el dataset
        print("Re-entrenando modelos de APM...")
        self.model_apm_p15 = self._train_quantile_model(X_apm, y_apm, 0.15, max_depth=4)
        self.model_apm_p50 = self._train_quantile_model(X_apm, y_apm, 0.50, max_depth=4)
        self.model_apm_p85 = self._train_quantile_model(X_apm, y_apm, 0.85, max_depth=4)

        # ============================================================
        # GUARDAR MODELOS (entrenados con TODO el dataset)
        # ============================================================
        print("\nGuardando modelos (entrenados con todos los datos)...")

        # Minutos (cuantiles)
        joblib.dump(self.model_minutes_p15, MODELS_PATH / "model_minutes_p15.joblib")
        joblib.dump(self.model_minutes_p50, MODELS_PATH / "model_minutes_p50.joblib")
        joblib.dump(self.model_minutes_p85, MODELS_PATH / "model_minutes_p85.joblib")

        # PPM (cuantiles)
        joblib.dump(self.model_ppm_p15, MODELS_PATH / "model_ppm_p15.joblib")
        joblib.dump(self.model_ppm_p50, MODELS_PATH / "model_ppm_p50.joblib")
        joblib.dump(self.model_ppm_p85, MODELS_PATH / "model_ppm_p85.joblib")

        # RPM (cuantiles)
        joblib.dump(self.model_rpm_p15, MODELS_PATH / "model_rpm_p15.joblib")
        joblib.dump(self.model_rpm_p50, MODELS_PATH / "model_rpm_p50.joblib")
        joblib.dump(self.model_rpm_p85, MODELS_PATH / "model_rpm_p85.joblib")

        # APM (cuantiles)
        joblib.dump(self.model_apm_p15, MODELS_PATH / "model_apm_p15.joblib")
        joblib.dump(self.model_apm_p50, MODELS_PATH / "model_apm_p50.joblib")
        joblib.dump(self.model_apm_p85, MODELS_PATH / "model_apm_p85.joblib")

        # Calibración y features
        joblib.dump(self.calibration_curve, MODELS_PATH / "calibration_curve.joblib")
        joblib.dump(self.probability_calibrator, MODELS_PATH / "probability_calibrator.joblib")
        joblib.dump(self.minutes_features, MODELS_PATH / "minutes_features.joblib")

        print("Modelos guardados correctamente.")

    def load_models(self):
        """Carga los modelos entrenados (con cuantiles)."""
        # Minutos (cuantiles)
        self.model_minutes_p15 = joblib.load(MODELS_PATH / "model_minutes_p15.joblib")
        self.model_minutes_p50 = joblib.load(MODELS_PATH / "model_minutes_p50.joblib")
        self.model_minutes_p85 = joblib.load(MODELS_PATH / "model_minutes_p85.joblib")

        # PPM (cuantiles)
        self.model_ppm_p15 = joblib.load(MODELS_PATH / "model_ppm_p15.joblib")
        self.model_ppm_p50 = joblib.load(MODELS_PATH / "model_ppm_p50.joblib")
        self.model_ppm_p85 = joblib.load(MODELS_PATH / "model_ppm_p85.joblib")

        # RPM (cuantiles)
        self.model_rpm_p15 = joblib.load(MODELS_PATH / "model_rpm_p15.joblib")
        self.model_rpm_p50 = joblib.load(MODELS_PATH / "model_rpm_p50.joblib")
        self.model_rpm_p85 = joblib.load(MODELS_PATH / "model_rpm_p85.joblib")

        # APM (cuantiles)
        self.model_apm_p15 = joblib.load(MODELS_PATH / "model_apm_p15.joblib")
        self.model_apm_p50 = joblib.load(MODELS_PATH / "model_apm_p50.joblib")
        self.model_apm_p85 = joblib.load(MODELS_PATH / "model_apm_p85.joblib")

        # Calibración y features
        try:
            self.calibration_curve = joblib.load(MODELS_PATH / "calibration_curve.joblib")
        except FileNotFoundError:
            self.calibration_curve = None

        try:
            self.probability_calibrator = joblib.load(MODELS_PATH / "probability_calibrator.joblib")
        except FileNotFoundError:
            self.probability_calibrator = ProbabilityCalibrator()

        self.minutes_features = joblib.load(MODELS_PATH / "minutes_features.joblib")

    def predict_player(self, player_name: str, opponent: str, is_home: bool = True,
                       days_rest: int = None, minutes_override: float = None) -> dict:
        """
        Predice estadísticas usando Minutos × Eficiencia con CUANTILES.

        Retorna P15 (piso), P50 (mediana), P85 (techo) para puntos, rebotes, asistencias.

        Args:
            days_rest: Si es None, se calcula automáticamente de los datos.
        """
        if self.model_minutes_p50 is None:
            self.load_models()

        conn = sqlite3.connect(DB_PATH)
        # LIMIT 82 = temporada completa (alineado con entrenamiento)
        player_df = pd.read_sql("""
            SELECT * FROM player_game_logs
            WHERE player_name = ?
            ORDER BY game_date DESC
            LIMIT 82
        """, conn, params=[player_name])
        conn.close()

        if player_df.empty:
            return {"error": f"No se encontró al jugador: {player_name}"}

        # Calcular eficiencias históricas
        player_df["ppm"] = player_df["pts"] / player_df["min"].replace(0, 1)
        player_df["rpm"] = player_df["reb"] / player_df["min"].replace(0, 1)
        player_df["apm"] = player_df["ast"] / player_df["min"].replace(0, 1)

        # Calcular days_rest desde fechas reales si no se proporciona
        if days_rest is None:
            player_df["game_date"] = pd.to_datetime(player_df["game_date"])
            if len(player_df) >= 2:
                # Días entre el partido más reciente y el anterior
                most_recent = player_df["game_date"].iloc[0]
                second_recent = player_df["game_date"].iloc[1]
                days_rest = (most_recent - second_recent).days
                days_rest = max(1, min(days_rest, 7))  # Clamp entre 1 y 7
            else:
                days_rest = 2  # Default

        last_5 = player_df.head(5)
        last_10 = player_df.head(10)
        vs_opp = player_df[player_df["opponent_abbrev"] == opponent]

        # Obtener opponent stats (DvP, pace, def_rating)
        opp_stats = get_opponent_stats()
        player_position = opp_stats.infer_position(player_df)
        dvp = opp_stats.get_dvp(opponent, player_position)
        team_stats = opp_stats.get_team_stats(opponent)

        opp_dvp_pts = dvp["pts_allowed"]
        opp_dvp_reb = dvp["reb_allowed"]
        opp_dvp_ast = dvp["ast_allowed"]
        opp_pace = team_stats["pace_factor"]
        opp_def_rating = team_stats["def_rating"]

        # Features para MINUTOS
        similar_games = player_df[player_df["is_home"] == (1 if is_home else 0)]

        min_features = {
            "avg_min_5": last_5["min"].mean(),
            "avg_min_10": last_10["min"].mean(),
            "season_avg_min": player_df["min"].mean(),
            "std_min": player_df["min"].std(),
            "min_trend": last_5["min"].mean() - last_10["min"].mean(),
            "min_consistency": 1 - (player_df["min"].std() / player_df["min"].mean()) if player_df["min"].mean() > 0 else 0,
            "days_rest": min(days_rest, 7),
            "is_b2b": 1 if days_rest <= 1 else 0,
            "is_home": 1 if is_home else 0,
            "avg_min_similar": similar_games["min"].mean() if len(similar_games) > 0 else player_df["min"].mean()
        }

        X_min = pd.DataFrame([min_features])[self.minutes_features].fillna(0)

        # Predecir MINUTOS (con cuantiles)
        if minutes_override is not None:
            pred_min_p15 = minutes_override
            pred_min_p50 = minutes_override
            pred_min_p85 = minutes_override
            minutes_source = "OVERRIDE"
        else:
            pred_min_p15 = self.model_minutes_p15.predict(X_min)[0]
            pred_min_p50 = self.model_minutes_p50.predict(X_min)[0]
            pred_min_p85 = self.model_minutes_p85.predict(X_min)[0]
            minutes_source = "MODEL"

        # Features para EFICIENCIA (incluye opponent stats)
        ppm_features = {
            "avg_ppm_5": last_5["ppm"].mean(),
            "avg_ppm_10": last_10["ppm"].mean(),
            "season_avg_ppm": player_df["ppm"].mean(),
            "std_ppm": player_df["ppm"].std(),
            "ppm_trend": last_5["ppm"].mean() - last_10["ppm"].mean(),
            "vs_opp_ppm": vs_opp["ppm"].mean() if len(vs_opp) > 0 else player_df["ppm"].mean(),
            "is_home": 1 if is_home else 0,
            # Opponent stats
            "opp_dvp_pts": opp_dvp_pts,
            "opp_dvp_reb": opp_dvp_reb,
            "opp_dvp_ast": opp_dvp_ast,
            "opp_pace": opp_pace,
            "opp_def_rating": opp_def_rating
        }

        rpm_features = {
            "avg_rpm_5": last_5["rpm"].mean(),
            "avg_rpm_10": last_10["rpm"].mean(),
            "season_avg_rpm": player_df["rpm"].mean(),
            "std_rpm": player_df["rpm"].std(),
            "rpm_trend": last_5["rpm"].mean() - last_10["rpm"].mean(),
            "vs_opp_rpm": vs_opp["rpm"].mean() if len(vs_opp) > 0 else player_df["rpm"].mean(),
            "is_home": 1 if is_home else 0,
            # Opponent stats
            "opp_dvp_pts": opp_dvp_pts,
            "opp_dvp_reb": opp_dvp_reb,
            "opp_dvp_ast": opp_dvp_ast,
            "opp_pace": opp_pace,
            "opp_def_rating": opp_def_rating
        }

        apm_features = {
            "avg_apm_5": last_5["apm"].mean(),
            "avg_apm_10": last_10["apm"].mean(),
            "season_avg_apm": player_df["apm"].mean(),
            "std_apm": player_df["apm"].std(),
            "apm_trend": last_5["apm"].mean() - last_10["apm"].mean(),
            "vs_opp_apm": vs_opp["apm"].mean() if len(vs_opp) > 0 else player_df["apm"].mean(),
            "is_home": 1 if is_home else 0,
            # Opponent stats
            "opp_dvp_pts": opp_dvp_pts,
            "opp_dvp_reb": opp_dvp_reb,
            "opp_dvp_ast": opp_dvp_ast,
            "opp_pace": opp_pace,
            "opp_def_rating": opp_def_rating
        }

        # Cada modelo usa sus propias features específicas
        ppm_cols = [
            "avg_ppm_5", "avg_ppm_10", "season_avg_ppm", "std_ppm",
            "ppm_trend", "vs_opp_ppm", "is_home"
        ]
        rpm_cols = [
            "avg_rpm_5", "avg_rpm_10", "season_avg_rpm", "std_rpm",
            "rpm_trend", "vs_opp_rpm", "is_home"
        ]
        apm_cols = [
            "avg_apm_5", "avg_apm_10", "season_avg_apm", "std_apm",
            "apm_trend", "vs_opp_apm", "is_home"
        ]

        X_ppm = pd.DataFrame([ppm_features])[ppm_cols].fillna(0)
        X_rpm = pd.DataFrame([rpm_features])[rpm_cols].fillna(0)
        X_apm = pd.DataFrame([apm_features])[apm_cols].fillna(0)

        # Predecir PPM (con cuantiles)
        pred_ppm_p15 = self.model_ppm_p15.predict(X_ppm)[0]
        pred_ppm_p50 = self.model_ppm_p50.predict(X_ppm)[0]
        pred_ppm_p85 = self.model_ppm_p85.predict(X_ppm)[0]

        # RPM (con cuantiles)
        pred_rpm_p15 = self.model_rpm_p15.predict(X_rpm)[0]
        pred_rpm_p50 = self.model_rpm_p50.predict(X_rpm)[0]
        pred_rpm_p85 = self.model_rpm_p85.predict(X_rpm)[0]

        # APM (con cuantiles)
        pred_apm_p15 = self.model_apm_p15.predict(X_apm)[0]
        pred_apm_p50 = self.model_apm_p50.predict(X_apm)[0]
        pred_apm_p85 = self.model_apm_p85.predict(X_apm)[0]

        # === PREDICCIÓN FINAL CON CUANTILES ===
        # P15 = min bajo × eficiencia baja (mal día)
        # P50 = min mediana × eficiencia mediana (día normal)
        # P85 = min alto × eficiencia alta (buen día)

        pred_pts_p15 = pred_min_p15 * pred_ppm_p15
        pred_pts_p50 = pred_min_p50 * pred_ppm_p50
        pred_pts_p85 = pred_min_p85 * pred_ppm_p85

        pred_reb_p15 = pred_min_p15 * pred_rpm_p15
        pred_reb_p50 = pred_min_p50 * pred_rpm_p50
        pred_reb_p85 = pred_min_p85 * pred_rpm_p85

        pred_ast_p15 = pred_min_p15 * pred_apm_p15
        pred_ast_p50 = pred_min_p50 * pred_apm_p50
        pred_ast_p85 = pred_min_p85 * pred_apm_p85

        return {
            "player": player_name,
            "opponent": opponent,
            "is_home": is_home,
            "days_rest": days_rest,
            "model_type": "QUANTILE REGRESSION",
            "matchup_info": {
                "player_position": player_position,
                "opp_dvp_pts": round(opp_dvp_pts, 1),
                "opp_dvp_reb": round(opp_dvp_reb, 1),
                "opp_dvp_ast": round(opp_dvp_ast, 1),
                "opp_pace": round(opp_pace, 3),
                "opp_def_rating": round(opp_def_rating, 3)
            },
            "minutes": {
                "p15_floor": round(pred_min_p15, 1),
                "p50_median": round(pred_min_p50, 1),
                "p85_ceiling": round(pred_min_p85, 1),
                "source": minutes_source,
                "avg_last_5": round(min_features["avg_min_5"], 1)
            },
            "efficiency_ppm": {
                "p15_floor": round(pred_ppm_p15, 3),
                "p50_median": round(pred_ppm_p50, 3),
                "p85_ceiling": round(pred_ppm_p85, 3)
            },
            "predictions_pts": {
                "p15_floor": round(pred_pts_p15, 1),
                "p50_median": round(pred_pts_p50, 1),
                "p85_ceiling": round(pred_pts_p85, 1),
                "range": f"{round(pred_pts_p15, 1)} - {round(pred_pts_p85, 1)}"
            },
            "predictions_reb": {
                "p15_floor": round(pred_reb_p15, 1),
                "p50_median": round(pred_reb_p50, 1),
                "p85_ceiling": round(pred_reb_p85, 1),
                "range": f"{round(pred_reb_p15, 1)} - {round(pred_reb_p85, 1)}"
            },
            "predictions_ast": {
                "p15_floor": round(pred_ast_p15, 1),
                "p50_median": round(pred_ast_p50, 1),
                "p85_ceiling": round(pred_ast_p85, 1),
                "range": f"{round(pred_ast_p15, 1)} - {round(pred_ast_p85, 1)}"
            },
            "predictions": {
                "pts": round(pred_pts_p50, 1),  # Mediana como principal
                "reb": round(pred_reb_p50, 1),
                "ast": round(pred_ast_p50, 1)
            },
            "formula": {
                "floor": f"{round(pred_min_p15, 1)} min × {round(pred_ppm_p15, 3)} = {round(pred_pts_p15, 1)} pts",
                "median": f"{round(pred_min_p50, 1)} min × {round(pred_ppm_p50, 3)} = {round(pred_pts_p50, 1)} pts",
                "ceiling": f"{round(pred_min_p85, 1)} min × {round(pred_ppm_p85, 3)} = {round(pred_pts_p85, 1)} pts"
            }
        }

    def calculate_probability_from_quantiles(self, p15: float, p50: float,
                                              p85: float, line: float,
                                              calibrate: bool = True) -> float:
        """
        Calcula probabilidad de Over usando cuantiles en lugar de distribución normal.

        Usa interpolación lineal entre cuantiles conocidos.
        Si calibrate=True, aplica calibración isotónica.

        Args:
            p15: Cuantil 15 (piso)
            p50: Cuantil 50 (mediana)
            p85: Cuantil 85 (techo)
            line: Línea de la apuesta
            calibrate: Si True, calibra la probabilidad con isotonic regression
        """
        # Si la línea está por debajo del piso, ~95%+ de Over
        if line <= p15:
            raw_prob = 0.92

        # Si la línea está por encima del techo, ~5% de Over
        elif line >= p85:
            raw_prob = 0.08

        # Si está entre P15 y P50
        elif line <= p50:
            # Interpolar entre 85% y 50%
            ratio = (line - p15) / (p50 - p15) if (p50 - p15) > 0 else 0.5
            raw_prob = 0.85 - (ratio * 0.35)  # 85% -> 50%

        # Si está entre P50 y P85
        else:
            # Interpolar entre 50% y 15%
            ratio = (line - p50) / (p85 - p50) if (p85 - p50) > 0 else 0.5
            raw_prob = 0.50 - (ratio * 0.35)  # 50% -> 15%

        # Aplicar calibración isotónica si está disponible
        if calibrate and self.probability_calibrator.is_fitted:
            return self.probability_calibrator.calibrate(raw_prob)

        return raw_prob

    def get_calibrated_probability(self, raw_prob: float) -> float:
        """
        Calibra una probabilidad cruda usando el modelo isotónico.

        Si el calibrador no está ajustado, devuelve la probabilidad original.
        """
        if self.probability_calibrator.is_fitted:
            return self.probability_calibrator.calibrate(raw_prob)
        return raw_prob

    def evaluate_bet(self, player_name: str, opponent: str, stat: str,
                     line: float, odds: float, is_home: bool = True,
                     minutes_override: float = None) -> dict:
        """
        Evalúa apuesta usando CUANTILES.

        LÓGICA CLAVE:
        - Si línea < P15 (piso): OVER de ALTO VALOR (incluso en mal día cubre)
        - Si línea > P85 (techo): UNDER de ALTO VALOR (incluso en buen día no llega)
        - Si línea entre P15-P50: OVER potencial
        - Si línea entre P50-P85: UNDER potencial
        """
        prediction = self.predict_player(
            player_name, opponent, is_home,
            minutes_override=minutes_override
        )

        if "error" in prediction:
            return prediction

        # Usar cuantiles reales para todas las estadísticas
        if stat == "pts":
            p15 = prediction["predictions_pts"]["p15_floor"]
            p50 = prediction["predictions_pts"]["p50_median"]
            p85 = prediction["predictions_pts"]["p85_ceiling"]
        elif stat == "reb":
            p15 = prediction["predictions_reb"]["p15_floor"]
            p50 = prediction["predictions_reb"]["p50_median"]
            p85 = prediction["predictions_reb"]["p85_ceiling"]
        elif stat == "ast":
            p15 = prediction["predictions_ast"]["p15_floor"]
            p50 = prediction["predictions_ast"]["p50_median"]
            p85 = prediction["predictions_ast"]["p85_ceiling"]
        else:
            # Fallback para stats no soportadas
            p50 = prediction["predictions"].get(stat, 10)
            p15 = p50 * 0.7
            p85 = p50 * 1.3

        prob_over = self.calculate_probability_from_quantiles(p15, p50, p85, line)

        prob_under = 1 - prob_over

        # Expected Value
        ev_over = (prob_over * odds) - 1
        ev_under = (prob_under * odds) - 1

        # Determinar señal de valor
        if line < p15:
            signal = "🔥 STRONG OVER"
            signal_reason = f"Línea ({line}) está DEBAJO del piso ({p15})"
        elif line > p85:
            signal = "🔥 STRONG UNDER"
            signal_reason = f"Línea ({line}) está ENCIMA del techo ({p85})"
        elif line < p50 and ev_over > 0.03:
            signal = "📈 LEAN OVER"
            signal_reason = f"Línea ({line}) debajo de mediana ({p50})"
        elif line > p50 and ev_under > 0.03:
            signal = "📉 LEAN UNDER"
            signal_reason = f"Línea ({line}) encima de mediana ({p50})"
        else:
            signal = "⚪ NO EDGE"
            signal_reason = "Línea cerca de la mediana, sin ventaja clara"

        # Recomendación final
        if ev_over > ev_under and ev_over > 0.05:
            recommendation = "OVER"
            best_ev = ev_over
        elif ev_under > 0.05:
            recommendation = "UNDER"
            best_ev = ev_under
        else:
            recommendation = "NO BET"
            best_ev = max(ev_over, ev_under)

        return {
            "player": player_name,
            "opponent": opponent,
            "stat": stat.upper(),
            "line": line,
            "odds": odds,
            "quantiles": {
                "p15_floor": p15,
                "p50_median": p50,
                "p85_ceiling": p85
            },
            "line_position": signal,
            "signal_reason": signal_reason,
            "prob_over": f"{prob_over * 100:.1f}%",
            "prob_under": f"{prob_under * 100:.1f}%",
            "ev_over": f"{ev_over * 100:.1f}%",
            "ev_under": f"{ev_under * 100:.1f}%",
            "recommendation": recommendation,
            "edge": f"{best_ev * 100:.1f}%",
            "confidence": "HIGH" if best_ev > 0.10 else "MEDIUM" if best_ev > 0.05 else "LOW"
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Modelo XGBoost (Minutos × Eficiencia)")
    parser.add_argument("--train", action="store_true", help="Entrenar modelos")
    parser.add_argument("--predict", type=str, help="Predecir jugador")
    parser.add_argument("--opponent", type=str, default="LAL", help="Oponente")
    parser.add_argument("--minutes", type=float, help="Override de minutos (si hay restricción)")

    args = parser.parse_args()

    predictor = NBAPropsPredictor()

    if args.train:
        predictor.train()
    elif args.predict:
        result = predictor.predict_player(
            args.predict, args.opponent,
            minutes_override=args.minutes
        )
        print("\n" + "="*50)
        print(f"PREDICCIÓN: {args.predict} vs {args.opponent}")
        print("="*50)
        for key, value in result.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
    else:
        print("Usa --train o --predict JUGADOR --opponent OPP")
