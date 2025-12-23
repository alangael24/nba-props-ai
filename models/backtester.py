"""
Walk-Forward Backtesting para NBA Props

Principio: El tiempo es unidireccional.
No podemos entrenar con datos del futuro para predecir el pasado.

Esquema:
1. Entrena con datos hasta fecha X
2. Predice partidos de fecha X+1 a X+7
3. Avanza la ventana y repite
4. Calcula ROI real sobre predicciones out-of-sample
"""

import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm

DB_PATH = Path(__file__).parent.parent / "data" / "nba_props.db"


class WalkForwardBacktester:
    """
    Backtesting con ventana rodante (walk-forward validation).

    Simula exactamente cómo usarías el modelo en producción:
    - Solo usa datos del PASADO para entrenar
    - Predice el FUTURO inmediato
    - Avanza en el tiempo y repite
    """

    def __init__(self, initial_bankroll: float = 1000.0,
                 bet_size: float = 50.0, min_ev_threshold: float = 0.05):
        """
        Args:
            initial_bankroll: Bankroll inicial para simular apuestas
            bet_size: Tamaño de cada apuesta (flat betting)
            min_ev_threshold: EV mínimo para apostar (5% default)
        """
        self.initial_bankroll = initial_bankroll
        self.bet_size = bet_size
        self.min_ev_threshold = min_ev_threshold

        # Resultados del backtest
        self.predictions = []
        self.bets = []
        self.bankroll_history = []

    def load_data(self) -> pd.DataFrame:
        """Carga todos los datos ordenados por fecha."""
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("""
            SELECT * FROM player_game_logs
            WHERE min > 10 AND player_name IS NOT NULL
            ORDER BY game_date
        """, conn)
        conn.close()

        df["game_date"] = pd.to_datetime(df["game_date"])
        return df

    def create_features_for_player(self, player_df: pd.DataFrame,
                                    current_idx: int) -> Dict:
        """
        Crea features para un jugador usando SOLO datos anteriores.
        """
        if current_idx < 10:
            return None

        historical = player_df.iloc[:current_idx]
        row = player_df.iloc[current_idx]

        last_5 = historical.tail(5)
        last_10 = historical.tail(10)

        # Calcular eficiencias
        historical_with_eff = historical.copy()
        historical_with_eff["ppm"] = historical_with_eff["pts"] / historical_with_eff["min"].replace(0, 1)

        opponent = row["opponent_abbrev"]
        vs_opp = historical[historical["opponent_abbrev"] == opponent]

        # Features de minutos
        min_features = {
            "avg_min_5": last_5["min"].mean(),
            "avg_min_10": last_10["min"].mean(),
            "season_avg_min": historical["min"].mean(),
            "std_min": historical["min"].std(),
            "min_trend": last_5["min"].mean() - last_10["min"].mean(),
            "min_consistency": 1 - (historical["min"].std() / historical["min"].mean()) if historical["min"].mean() > 0 else 0,
            "days_rest": 2,  # Simplificado
            "is_b2b": 0,
            "is_home": row["is_home"],
            "avg_min_similar": historical[historical["is_home"] == row["is_home"]]["min"].mean() if len(historical[historical["is_home"] == row["is_home"]]) > 0 else historical["min"].mean()
        }

        # Features de eficiencia
        ppm_features = {
            "avg_ppm_5": last_5["pts"].mean() / last_5["min"].replace(0, 1).mean() if last_5["min"].mean() > 0 else 0,
            "avg_ppm_10": last_10["pts"].mean() / last_10["min"].replace(0, 1).mean() if last_10["min"].mean() > 0 else 0,
            "season_avg_ppm": historical["pts"].sum() / historical["min"].replace(0, 1).sum() if historical["min"].sum() > 0 else 0,
            "std_ppm": (historical["pts"] / historical["min"].replace(0, 1)).std(),
            "ppm_trend": 0,
            "vs_opp_ppm": vs_opp["pts"].sum() / vs_opp["min"].replace(0, 1).sum() if len(vs_opp) > 0 and vs_opp["min"].sum() > 0 else historical["pts"].sum() / historical["min"].replace(0, 1).sum(),
            "is_home": row["is_home"]
        }

        return {
            "min_features": min_features,
            "ppm_features": ppm_features,
            "actual_pts": row["pts"],
            "actual_min": row["min"],
            "player_name": row["player_name"],
            "game_date": row["game_date"],
            "opponent": opponent
        }

    def train_models_up_to_date(self, df: pd.DataFrame,
                                 cutoff_date: datetime) -> Tuple:
        """
        Entrena modelos usando SOLO datos anteriores a cutoff_date.
        """
        train_df = df[df["game_date"] < cutoff_date].copy()

        if len(train_df) < 100:
            return None, None

        # Crear features de entrenamiento
        train_df["ppm"] = train_df["pts"] / train_df["min"].replace(0, 1)

        features_list = []
        for player_id in train_df["player_id"].unique():
            player_df = train_df[train_df["player_id"] == player_id].copy()
            player_df = player_df.sort_values("game_date").reset_index(drop=True)

            for i in range(10, len(player_df)):
                feat = self.create_features_for_player(player_df, i)
                if feat:
                    features_list.append(feat)

        if len(features_list) < 50:
            return None, None

        # Preparar datos
        X_min = pd.DataFrame([f["min_features"] for f in features_list])
        y_min = pd.Series([f["actual_min"] for f in features_list])

        X_ppm = pd.DataFrame([f["ppm_features"] for f in features_list])
        y_ppm = pd.Series([f["actual_pts"] / max(f["actual_min"], 1) for f in features_list])

        # Entrenar modelos de cuantiles
        min_features_cols = list(X_min.columns)
        ppm_features_cols = list(X_ppm.columns)

        # Modelo de minutos P50
        model_min = xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            objective='reg:quantileerror', quantile_alpha=0.50, random_state=42
        )
        model_min.fit(X_min.fillna(0), y_min)

        # Modelos de PPM (P15, P50, P85)
        model_ppm_p15 = xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            objective='reg:quantileerror', quantile_alpha=0.15, random_state=42
        )
        model_ppm_p50 = xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            objective='reg:quantileerror', quantile_alpha=0.50, random_state=42
        )
        model_ppm_p85 = xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            objective='reg:quantileerror', quantile_alpha=0.85, random_state=42
        )

        model_ppm_p15.fit(X_ppm.fillna(0), y_ppm)
        model_ppm_p50.fit(X_ppm.fillna(0), y_ppm)
        model_ppm_p85.fit(X_ppm.fillna(0), y_ppm)

        models = {
            "min": model_min,
            "ppm_p15": model_ppm_p15,
            "ppm_p50": model_ppm_p50,
            "ppm_p85": model_ppm_p85,
            "min_features": min_features_cols,
            "ppm_features": ppm_features_cols
        }

        return models, train_df

    def predict_game(self, models: Dict, player_df: pd.DataFrame,
                     idx: int, line: float = None) -> Dict:
        """
        Predice un partido específico.
        """
        feat = self.create_features_for_player(player_df, idx)
        if feat is None:
            return None

        X_min = pd.DataFrame([feat["min_features"]])[models["min_features"]].fillna(0)
        X_ppm = pd.DataFrame([feat["ppm_features"]])[models["ppm_features"]].fillna(0)

        pred_min = models["min"].predict(X_min)[0]
        pred_ppm_p15 = models["ppm_p15"].predict(X_ppm)[0]
        pred_ppm_p50 = models["ppm_p50"].predict(X_ppm)[0]
        pred_ppm_p85 = models["ppm_p85"].predict(X_ppm)[0]

        # Predicciones de puntos
        pred_pts_p15 = pred_min * pred_ppm_p15
        pred_pts_p50 = pred_min * pred_ppm_p50
        pred_pts_p85 = pred_min * pred_ppm_p85

        actual_pts = feat["actual_pts"]

        # Si no hay línea, usar la mediana como línea "justa"
        if line is None:
            line = pred_pts_p50

        # Calcular probabilidad y EV
        if line <= pred_pts_p15:
            prob_over = 0.92
        elif line >= pred_pts_p85:
            prob_over = 0.08
        elif line <= pred_pts_p50:
            ratio = (line - pred_pts_p15) / (pred_pts_p50 - pred_pts_p15) if (pred_pts_p50 - pred_pts_p15) > 0 else 0.5
            prob_over = 0.85 - (ratio * 0.35)
        else:
            ratio = (line - pred_pts_p50) / (pred_pts_p85 - pred_pts_p50) if (pred_pts_p85 - pred_pts_p50) > 0 else 0.5
            prob_over = 0.50 - (ratio * 0.35)

        odds = 1.90  # Cuota estándar
        ev_over = (prob_over * odds) - 1
        ev_under = ((1 - prob_over) * odds) - 1

        return {
            "player": feat["player_name"],
            "game_date": feat["game_date"],
            "opponent": feat["opponent"],
            "pred_pts_p15": pred_pts_p15,
            "pred_pts_p50": pred_pts_p50,
            "pred_pts_p85": pred_pts_p85,
            "actual_pts": actual_pts,
            "line": line,
            "prob_over": prob_over,
            "ev_over": ev_over,
            "ev_under": ev_under,
            "bet_recommendation": "OVER" if ev_over > self.min_ev_threshold else ("UNDER" if ev_under > self.min_ev_threshold else "NO_BET"),
            "actual_over": actual_pts > line
        }

    def run_backtest(self, start_date: str = "2023-06-01",
                     end_date: str = "2024-12-01",
                     retrain_frequency_days: int = 14) -> Dict:
        """
        Ejecuta backtest walk-forward.

        Args:
            start_date: Fecha de inicio del backtest
            end_date: Fecha de fin
            retrain_frequency_days: Cada cuántos días re-entrenar el modelo
        """
        print("="*60)
        print("WALK-FORWARD BACKTEST")
        print("="*60)
        print(f"Período: {start_date} a {end_date}")
        print(f"Re-entrenamiento cada: {retrain_frequency_days} días")
        print(f"Bankroll inicial: ${self.initial_bankroll}")
        print(f"Tamaño apuesta: ${self.bet_size}")
        print(f"EV mínimo: {self.min_ev_threshold*100}%")

        df = self.load_data()
        print(f"\nDatos cargados: {len(df)} registros")

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        current_bankroll = self.initial_bankroll
        self.bankroll_history = [(start_dt, current_bankroll)]

        all_predictions = []
        all_bets = []

        current_date = start_dt
        models = None
        last_train_date = None

        # Iterar día por día
        date_range = pd.date_range(start_dt, end_dt, freq='D')

        for current_date in tqdm(date_range, desc="Backtesting"):
            # Re-entrenar si es necesario
            if models is None or (current_date - last_train_date).days >= retrain_frequency_days:
                models, _ = self.train_models_up_to_date(df, current_date)
                last_train_date = current_date

                if models is None:
                    continue

            # Obtener partidos del día
            day_games = df[df["game_date"].dt.date == current_date.date()]

            if len(day_games) == 0:
                continue

            # Predecir cada partido
            for _, game in day_games.iterrows():
                player_df = df[df["player_id"] == game["player_id"]].copy()
                player_df = player_df[player_df["game_date"] <= current_date]
                player_df = player_df.sort_values("game_date").reset_index(drop=True)

                if len(player_df) < 11:
                    continue

                # El último índice es el partido actual
                idx = len(player_df) - 1

                # Simular línea (usamos mediana del jugador - 0.5 como línea típica)
                historical_median = player_df.iloc[:-1]["pts"].median()
                simulated_line = historical_median - 0.5

                pred = self.predict_game(models, player_df, idx, line=simulated_line)

                if pred is None:
                    continue

                all_predictions.append(pred)

                # Simular apuesta si hay valor
                if pred["bet_recommendation"] != "NO_BET" and current_bankroll >= self.bet_size:
                    bet_type = pred["bet_recommendation"]
                    won = (bet_type == "OVER" and pred["actual_over"]) or \
                          (bet_type == "UNDER" and not pred["actual_over"])

                    if won:
                        profit = self.bet_size * 0.90  # Cuota 1.90 - stake
                    else:
                        profit = -self.bet_size

                    current_bankroll += profit

                    bet_record = {
                        **pred,
                        "bet_type": bet_type,
                        "stake": self.bet_size,
                        "profit": profit,
                        "won": won,
                        "bankroll_after": current_bankroll
                    }
                    all_bets.append(bet_record)
                    self.bankroll_history.append((current_date, current_bankroll))

        self.predictions = all_predictions
        self.bets = all_bets

        return self.calculate_results()

    def calculate_results(self) -> Dict:
        """Calcula métricas finales del backtest."""
        if not self.bets:
            return {"error": "No se realizaron apuestas"}

        bets_df = pd.DataFrame(self.bets)

        total_bets = len(bets_df)
        wins = bets_df["won"].sum()
        losses = total_bets - wins
        win_rate = wins / total_bets

        total_profit = bets_df["profit"].sum()
        roi = total_profit / (total_bets * self.bet_size) * 100

        final_bankroll = bets_df["bankroll_after"].iloc[-1]

        # Por tipo de apuesta
        over_bets = bets_df[bets_df["bet_type"] == "OVER"]
        under_bets = bets_df[bets_df["bet_type"] == "UNDER"]

        results = {
            "total_predictions": len(self.predictions),
            "total_bets": total_bets,
            "wins": wins,
            "losses": losses,
            "win_rate": f"{win_rate*100:.1f}%",
            "total_profit": f"${total_profit:.2f}",
            "roi": f"{roi:.2f}%",
            "initial_bankroll": f"${self.initial_bankroll:.2f}",
            "final_bankroll": f"${final_bankroll:.2f}",
            "bankroll_growth": f"{((final_bankroll/self.initial_bankroll)-1)*100:.1f}%",
            "over_bets": {
                "total": len(over_bets),
                "wins": over_bets["won"].sum() if len(over_bets) > 0 else 0,
                "win_rate": f"{over_bets['won'].mean()*100:.1f}%" if len(over_bets) > 0 else "N/A"
            },
            "under_bets": {
                "total": len(under_bets),
                "wins": under_bets["won"].sum() if len(under_bets) > 0 else 0,
                "win_rate": f"{under_bets['won'].mean()*100:.1f}%" if len(under_bets) > 0 else "N/A"
            }
        }

        return results

    def print_results(self):
        """Imprime resultados del backtest."""
        results = self.calculate_results()

        print("\n" + "="*60)
        print("RESULTADOS DEL BACKTEST")
        print("="*60)

        for key, value in results.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")

        # Análisis de predicciones
        if self.predictions:
            pred_df = pd.DataFrame(self.predictions)
            mae = abs(pred_df["pred_pts_p50"] - pred_df["actual_pts"]).mean()
            print(f"\nMAE predicciones: {mae:.2f} puntos")

            # Calibración de cuantiles
            below_p15 = (pred_df["actual_pts"] < pred_df["pred_pts_p15"]).mean()
            below_p50 = (pred_df["actual_pts"] < pred_df["pred_pts_p50"]).mean()
            below_p85 = (pred_df["actual_pts"] < pred_df["pred_pts_p85"]).mean()

            print(f"\nCalibración de cuantiles:")
            print(f"  Actual < P15: {below_p15*100:.1f}% (esperado ~15%)")
            print(f"  Actual < P50: {below_p50*100:.1f}% (esperado ~50%)")
            print(f"  Actual < P85: {below_p85*100:.1f}% (esperado ~85%)")


def run_backtest():
    """Función principal para ejecutar backtest."""
    backtester = WalkForwardBacktester(
        initial_bankroll=1000.0,
        bet_size=50.0,
        min_ev_threshold=0.05
    )

    results = backtester.run_backtest(
        start_date="2023-10-01",  # Inicio temporada 2023-24
        end_date="2024-04-01",    # Fin temporada regular
        retrain_frequency_days=14
    )

    backtester.print_results()

    return backtester


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Walk-Forward Backtest")
    parser.add_argument("--start", type=str, default="2023-10-01", help="Fecha inicio")
    parser.add_argument("--end", type=str, default="2024-04-01", help="Fecha fin")
    parser.add_argument("--bankroll", type=float, default=1000.0, help="Bankroll inicial")
    parser.add_argument("--bet-size", type=float, default=50.0, help="Tamaño apuesta")
    parser.add_argument("--min-ev", type=float, default=0.05, help="EV mínimo")
    parser.add_argument("--retrain-days", type=int, default=14, help="Días entre re-entrenamientos")

    args = parser.parse_args()

    backtester = WalkForwardBacktester(
        initial_bankroll=args.bankroll,
        bet_size=args.bet_size,
        min_ev_threshold=args.min_ev
    )

    backtester.run_backtest(
        start_date=args.start,
        end_date=args.end,
        retrain_frequency_days=args.retrain_days
    )

    backtester.print_results()
