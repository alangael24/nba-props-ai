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
import xgboost as xgb

DB_PATH = Path(__file__).parent.parent / "data" / "nba_props.db"
MODELS_PATH = Path(__file__).parent


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

        # Modelos RPM y APM (mediana solamente para simplicidad)
        self.model_rpm = None
        self.model_apm = None

        self.minutes_features = None
        self.efficiency_features = None

        # Cuantiles que usamos
        self.quantiles = [0.15, 0.50, 0.85]

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features separadas para Minutos y Eficiencia.
        """
        df = df.copy()
        df = df.sort_values(["player_id", "game_date"])
        df["game_date"] = pd.to_datetime(df["game_date"])

        # Calcular eficiencias (por minuto)
        df["ppm"] = df["pts"] / df["min"].replace(0, 1)  # Points per minute
        df["rpm"] = df["reb"] / df["min"].replace(0, 1)  # Rebounds per minute
        df["apm"] = df["ast"] / df["min"].replace(0, 1)  # Assists per minute

        features = []

        for player_id in df["player_id"].unique():
            player_df = df[df["player_id"] == player_id].copy()

            if len(player_df) < 10:
                continue

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
        """Features para predecir EFICIENCIA."""
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
            "is_home"
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

    def train(self, test_size: float = 0.2):
        """
        Entrena modelos con REGRESIÓN DE CUANTILES.

        Para cada variable predecimos:
        - P15: El piso (mal día)
        - P50: La mediana (día normal)
        - P85: El techo (buen día)
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

        split_idx = int(len(features_df) * (1 - test_size))

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

        X_min_train = X_min.iloc[:split_idx]
        X_min_test = X_min.iloc[split_idx:]
        y_min_train = y_min.iloc[:split_idx]
        y_min_test = y_min.iloc[split_idx:]

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

        X_ppm_train = X_ppm.iloc[:split_idx]
        y_ppm_train = y_ppm.iloc[:split_idx]
        X_ppm_test = X_ppm.iloc[split_idx:]
        y_ppm_test = y_ppm.iloc[split_idx:]

        print("Entrenando PPM cuantiles...")
        self.model_ppm_p15 = self._train_quantile_model(X_ppm_train, y_ppm_train, 0.15, max_depth=4)
        self.model_ppm_p50 = self._train_quantile_model(X_ppm_train, y_ppm_train, 0.50, max_depth=4)
        self.model_ppm_p85 = self._train_quantile_model(X_ppm_train, y_ppm_train, 0.85, max_depth=4)

        pred_ppm_p50 = self.model_ppm_p50.predict(X_ppm_test)
        mae_ppm = mean_absolute_error(y_ppm_test, pred_ppm_p50)
        print(f"MAE PPM (mediana): {mae_ppm:.3f}")

        # ============================================================
        # MODELOS RPM y APM (solo mediana por simplicidad)
        # ============================================================
        print("\n" + "="*50)
        print("MODELOS RPM y APM (mediana)")
        print("="*50)

        rpm_features = [
            "avg_rpm_5", "avg_rpm_10", "season_avg_rpm", "std_rpm",
            "rpm_trend", "vs_opp_rpm", "is_home"
        ]
        X_rpm = features_df[rpm_features].fillna(0)
        y_rpm = features_df["actual_rpm"]

        self.model_rpm = self._train_quantile_model(
            X_rpm.iloc[:split_idx], y_rpm.iloc[:split_idx], 0.50, max_depth=4
        )
        pred_rpm = self.model_rpm.predict(X_rpm.iloc[split_idx:])
        mae_rpm = mean_absolute_error(y_rpm.iloc[split_idx:], pred_rpm)
        print(f"MAE RPM: {mae_rpm:.3f}")

        apm_features = [
            "avg_apm_5", "avg_apm_10", "season_avg_apm", "std_apm",
            "apm_trend", "vs_opp_apm", "is_home"
        ]
        X_apm = features_df[apm_features].fillna(0)
        y_apm = features_df["actual_apm"]

        self.model_apm = self._train_quantile_model(
            X_apm.iloc[:split_idx], y_apm.iloc[:split_idx], 0.50, max_depth=4
        )
        pred_apm = self.model_apm.predict(X_apm.iloc[split_idx:])
        mae_apm = mean_absolute_error(y_apm.iloc[split_idx:], pred_apm)
        print(f"MAE APM: {mae_apm:.3f}")

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

        actual_pts = features_df["actual_pts"].iloc[split_idx:]

        mae_pts = mean_absolute_error(actual_pts, final_pred_pts_p50)
        print(f"MAE Puntos (mediana): {mae_pts:.2f}")

        # Verificar cobertura de cuantiles en puntos finales
        pts_above_p15 = (actual_pts.values > final_pred_pts_p15).mean()
        pts_below_p85 = (actual_pts.values < final_pred_pts_p85).mean()
        print(f"Puntos > P15: {pts_above_p15*100:.1f}% (esperado ~85%)")
        print(f"Puntos < P85: {pts_below_p85*100:.1f}% (esperado ~85%)")

        # Ejemplo de rango
        print(f"\nEjemplo de predicción con rango:")
        print(f"  P15 (piso):    {final_pred_pts_p15.mean():.1f} pts")
        print(f"  P50 (mediana): {final_pred_pts_p50.mean():.1f} pts")
        print(f"  P85 (techo):   {final_pred_pts_p85.mean():.1f} pts")

        # ============================================================
        # GUARDAR MODELOS
        # ============================================================
        print("\nGuardando modelos...")

        # Minutos (cuantiles)
        joblib.dump(self.model_minutes_p15, MODELS_PATH / "model_minutes_p15.joblib")
        joblib.dump(self.model_minutes_p50, MODELS_PATH / "model_minutes_p50.joblib")
        joblib.dump(self.model_minutes_p85, MODELS_PATH / "model_minutes_p85.joblib")

        # PPM (cuantiles)
        joblib.dump(self.model_ppm_p15, MODELS_PATH / "model_ppm_p15.joblib")
        joblib.dump(self.model_ppm_p50, MODELS_PATH / "model_ppm_p50.joblib")
        joblib.dump(self.model_ppm_p85, MODELS_PATH / "model_ppm_p85.joblib")

        # RPM y APM
        joblib.dump(self.model_rpm, MODELS_PATH / "model_rpm.joblib")
        joblib.dump(self.model_apm, MODELS_PATH / "model_apm.joblib")

        # Features
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

        # RPM y APM
        self.model_rpm = joblib.load(MODELS_PATH / "model_rpm.joblib")
        self.model_apm = joblib.load(MODELS_PATH / "model_apm.joblib")

        # Features
        self.minutes_features = joblib.load(MODELS_PATH / "minutes_features.joblib")

    def predict_player(self, player_name: str, opponent: str, is_home: bool = True,
                       days_rest: int = 2, minutes_override: float = None) -> dict:
        """
        Predice estadísticas usando Minutos × Eficiencia con CUANTILES.

        Retorna P15 (piso), P50 (mediana), P85 (techo) para puntos.
        """
        if self.model_minutes_p50 is None:
            self.load_models()

        conn = sqlite3.connect(DB_PATH)
        player_df = pd.read_sql("""
            SELECT * FROM player_game_logs
            WHERE player_name = ?
            ORDER BY game_date DESC
            LIMIT 30
        """, conn, params=[player_name])
        conn.close()

        if player_df.empty:
            return {"error": f"No se encontró al jugador: {player_name}"}

        # Calcular eficiencias históricas
        player_df["ppm"] = player_df["pts"] / player_df["min"].replace(0, 1)
        player_df["rpm"] = player_df["reb"] / player_df["min"].replace(0, 1)
        player_df["apm"] = player_df["ast"] / player_df["min"].replace(0, 1)

        last_5 = player_df.head(5)
        last_10 = player_df.head(10)
        vs_opp = player_df[player_df["opponent_abbrev"] == opponent]

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

        # Features para EFICIENCIA
        ppm_features = {
            "avg_ppm_5": last_5["ppm"].mean(),
            "avg_ppm_10": last_10["ppm"].mean(),
            "season_avg_ppm": player_df["ppm"].mean(),
            "std_ppm": player_df["ppm"].std(),
            "ppm_trend": last_5["ppm"].mean() - last_10["ppm"].mean(),
            "vs_opp_ppm": vs_opp["ppm"].mean() if len(vs_opp) > 0 else player_df["ppm"].mean(),
            "is_home": 1 if is_home else 0
        }

        rpm_features = {
            "avg_rpm_5": last_5["rpm"].mean(),
            "avg_rpm_10": last_10["rpm"].mean(),
            "season_avg_rpm": player_df["rpm"].mean(),
            "std_rpm": player_df["rpm"].std(),
            "rpm_trend": last_5["rpm"].mean() - last_10["rpm"].mean(),
            "vs_opp_rpm": vs_opp["rpm"].mean() if len(vs_opp) > 0 else player_df["rpm"].mean(),
            "is_home": 1 if is_home else 0
        }

        apm_features = {
            "avg_apm_5": last_5["apm"].mean(),
            "avg_apm_10": last_10["apm"].mean(),
            "season_avg_apm": player_df["apm"].mean(),
            "std_apm": player_df["apm"].std(),
            "apm_trend": last_5["apm"].mean() - last_10["apm"].mean(),
            "vs_opp_apm": vs_opp["apm"].mean() if len(vs_opp) > 0 else player_df["apm"].mean(),
            "is_home": 1 if is_home else 0
        }

        X_ppm = pd.DataFrame([ppm_features]).fillna(0)
        X_rpm = pd.DataFrame([rpm_features]).fillna(0)
        X_apm = pd.DataFrame([apm_features]).fillna(0)

        # Predecir PPM (con cuantiles)
        pred_ppm_p15 = self.model_ppm_p15.predict(X_ppm)[0]
        pred_ppm_p50 = self.model_ppm_p50.predict(X_ppm)[0]
        pred_ppm_p85 = self.model_ppm_p85.predict(X_ppm)[0]

        # RPM y APM (solo mediana)
        pred_rpm = self.model_rpm.predict(X_rpm)[0]
        pred_apm = self.model_apm.predict(X_apm)[0]

        # === PREDICCIÓN FINAL CON CUANTILES ===
        # P15 = min bajo × ppm bajo (mal día)
        pred_pts_p15 = pred_min_p15 * pred_ppm_p15
        # P50 = min mediana × ppm mediana (día normal)
        pred_pts_p50 = pred_min_p50 * pred_ppm_p50
        # P85 = min alto × ppm alto (buen día)
        pred_pts_p85 = pred_min_p85 * pred_ppm_p85

        pred_reb = pred_min_p50 * pred_rpm
        pred_ast = pred_min_p50 * pred_apm

        return {
            "player": player_name,
            "opponent": opponent,
            "is_home": is_home,
            "model_type": "QUANTILE REGRESSION",
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
            "predictions": {
                "pts": round(pred_pts_p50, 1),  # Mediana como principal
                "reb": round(pred_reb, 1),
                "ast": round(pred_ast, 1)
            },
            "formula": {
                "floor": f"{round(pred_min_p15, 1)} min × {round(pred_ppm_p15, 3)} = {round(pred_pts_p15, 1)} pts",
                "median": f"{round(pred_min_p50, 1)} min × {round(pred_ppm_p50, 3)} = {round(pred_pts_p50, 1)} pts",
                "ceiling": f"{round(pred_min_p85, 1)} min × {round(pred_ppm_p85, 3)} = {round(pred_pts_p85, 1)} pts"
            }
        }

    def calculate_probability_from_quantiles(self, p15: float, p50: float,
                                              p85: float, line: float) -> float:
        """
        Calcula probabilidad de Over usando cuantiles en lugar de distribución normal.

        Usa interpolación lineal entre cuantiles conocidos.
        """
        # Si la línea está por debajo del piso, ~95%+ de Over
        if line <= p15:
            return 0.92

        # Si la línea está por encima del techo, ~5% de Over
        if line >= p85:
            return 0.08

        # Si está entre P15 y P50
        if line <= p50:
            # Interpolar entre 85% y 50%
            ratio = (line - p15) / (p50 - p15) if (p50 - p15) > 0 else 0.5
            return 0.85 - (ratio * 0.35)  # 85% -> 50%

        # Si está entre P50 y P85
        else:
            # Interpolar entre 50% y 15%
            ratio = (line - p50) / (p85 - p50) if (p85 - p50) > 0 else 0.5
            return 0.50 - (ratio * 0.35)  # 50% -> 15%

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

        # Solo para puntos usamos cuantiles completos
        if stat == "pts":
            p15 = prediction["predictions_pts"]["p15_floor"]
            p50 = prediction["predictions_pts"]["p50_median"]
            p85 = prediction["predictions_pts"]["p85_ceiling"]

            prob_over = self.calculate_probability_from_quantiles(p15, p50, p85, line)
        else:
            # Para reb/ast usamos solo mediana con aproximación
            p50 = prediction["predictions"][stat]
            # Aproximación simple para reb/ast
            prob_over = 0.5 + (p50 - line) * 0.1
            prob_over = max(0.1, min(0.9, prob_over))
            p15 = p50 * 0.7
            p85 = p50 * 1.3

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
