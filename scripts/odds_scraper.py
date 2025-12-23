"""
Fase 4: Cazador de Valor
Obtiene cuotas de apuestas y calcula Expected Value.

Fuentes de cuotas:
- The Odds API (gratuita con límites)
- Scraping directo (con cuidado)
"""

import os
import json
from datetime import datetime
from typing import Optional
import httpx

# API Key para The Odds API (gratuita: 500 requests/mes)
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"


class OddsFetcher:
    """Obtiene cuotas de apuestas de múltiples fuentes."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or ODDS_API_KEY
        self.client = httpx.Client(timeout=30.0)

    def get_nba_games(self) -> list:
        """Obtiene los partidos de NBA disponibles para apuestas."""
        if not self.api_key:
            return self._get_mock_games()

        url = f"{ODDS_API_BASE}/sports/basketball_nba/odds"
        params = {
            "apiKey": self.api_key,
            "regions": "us",
            "markets": "player_points,player_rebounds,player_assists",
            "oddsFormat": "decimal"
        }

        try:
            response = self.client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error obteniendo cuotas: {e}")
            return self._get_mock_games()

    def get_player_props(self, player_name: str = None) -> list:
        """
        Obtiene props de jugadores.

        En producción, usar:
        - The Odds API (markets: player_points, player_rebounds, etc.)
        - DraftKings API
        - FanDuel scraping
        """
        if not self.api_key:
            return self._get_mock_player_props(player_name)

        url = f"{ODDS_API_BASE}/sports/basketball_nba/events"
        params = {
            "apiKey": self.api_key,
        }

        try:
            # Primero obtener eventos
            response = self.client.get(url, params=params)
            events = response.json()

            all_props = []
            for event in events[:5]:  # Limitar para no agotar API
                event_id = event.get("id")
                props = self._get_event_props(event_id)
                all_props.extend(props)

            return all_props

        except Exception as e:
            print(f"Error: {e}")
            return self._get_mock_player_props(player_name)

    def _get_event_props(self, event_id: str) -> list:
        """Obtiene props de un evento específico."""
        url = f"{ODDS_API_BASE}/sports/basketball_nba/events/{event_id}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": "us",
            "markets": "player_points,player_rebounds,player_assists",
            "oddsFormat": "decimal"
        }

        try:
            response = self.client.get(url, params=params)
            return response.json()
        except:
            return []

    def _get_mock_games(self) -> list:
        """Datos mock para desarrollo sin API key."""
        return [
            {
                "id": "mock_game_1",
                "sport_key": "basketball_nba",
                "home_team": "Los Angeles Lakers",
                "away_team": "Golden State Warriors",
                "commence_time": datetime.now().isoformat()
            },
            {
                "id": "mock_game_2",
                "sport_key": "basketball_nba",
                "home_team": "Boston Celtics",
                "away_team": "Milwaukee Bucks",
                "commence_time": datetime.now().isoformat()
            }
        ]

    def _get_mock_player_props(self, player_name: str = None) -> list:
        """Props mock para desarrollo."""
        mock_props = [
            {
                "player": "LeBron James",
                "team": "LAL",
                "opponent": "GSW",
                "props": {
                    "pts": {"line": 25.5, "over_odds": 1.91, "under_odds": 1.91},
                    "reb": {"line": 7.5, "over_odds": 1.87, "under_odds": 1.95},
                    "ast": {"line": 8.5, "over_odds": 1.90, "under_odds": 1.90}
                }
            },
            {
                "player": "Stephen Curry",
                "team": "GSW",
                "opponent": "LAL",
                "props": {
                    "pts": {"line": 28.5, "over_odds": 1.95, "under_odds": 1.87},
                    "reb": {"line": 5.5, "over_odds": 1.90, "under_odds": 1.90},
                    "ast": {"line": 5.5, "over_odds": 1.85, "under_odds": 1.97}
                }
            },
            {
                "player": "Jayson Tatum",
                "team": "BOS",
                "opponent": "MIL",
                "props": {
                    "pts": {"line": 27.5, "over_odds": 1.90, "under_odds": 1.90},
                    "reb": {"line": 8.5, "over_odds": 1.91, "under_odds": 1.91},
                    "ast": {"line": 4.5, "over_odds": 1.87, "under_odds": 1.95}
                }
            },
            {
                "player": "Giannis Antetokounmpo",
                "team": "MIL",
                "opponent": "BOS",
                "props": {
                    "pts": {"line": 31.5, "over_odds": 1.90, "under_odds": 1.90},
                    "reb": {"line": 11.5, "over_odds": 1.87, "under_odds": 1.95},
                    "ast": {"line": 5.5, "over_odds": 1.91, "under_odds": 1.91}
                }
            }
        ]

        if player_name:
            return [p for p in mock_props if player_name.lower() in p["player"].lower()]

        return mock_props


class EVCalculator:
    """Calcula Expected Value para apuestas."""

    @staticmethod
    def calculate_ev(probability: float, odds: float) -> float:
        """
        Calcula el Expected Value de una apuesta.

        EV = (Probabilidad * Cuota) - 1

        Args:
            probability: Probabilidad estimada (0-1)
            odds: Cuota decimal (ej: 1.90)

        Returns:
            EV como porcentaje (-1 a +infinito)
        """
        return (probability * odds) - 1

    @staticmethod
    def implied_probability(odds: float) -> float:
        """
        Calcula la probabilidad implícita de una cuota.

        P = 1 / odds
        """
        return 1 / odds

    @staticmethod
    def find_value_bets(props: list, predictions: dict,
                        min_ev: float = 0.05) -> list:
        """
        Encuentra apuestas con valor positivo.

        Args:
            props: Lista de props con cuotas
            predictions: Dict con predicciones {player: {stat: {prediction, probability}}}
            min_ev: EV mínimo para considerar (default 5%)

        Returns:
            Lista de apuestas con valor
        """
        value_bets = []

        for prop in props:
            player = prop["player"]

            if player not in predictions:
                continue

            player_preds = predictions[player]

            for stat, prop_data in prop["props"].items():
                if stat not in player_preds:
                    continue

                pred = player_preds[stat]
                line = prop_data["line"]
                over_odds = prop_data["over_odds"]
                under_odds = prop_data["under_odds"]

                # Probabilidad de Over según nuestro modelo
                prob_over = pred.get("prob_over", 0.5)
                prob_under = 1 - prob_over

                # Calcular EV
                ev_over = EVCalculator.calculate_ev(prob_over, over_odds)
                ev_under = EVCalculator.calculate_ev(prob_under, under_odds)

                # Probabilidad implícita del mercado
                implied_over = EVCalculator.implied_probability(over_odds)
                implied_under = EVCalculator.implied_probability(under_odds)

                # Si hay valor, agregar a la lista
                if ev_over >= min_ev:
                    value_bets.append({
                        "player": player,
                        "opponent": prop["opponent"],
                        "stat": stat,
                        "bet_type": "OVER",
                        "line": line,
                        "odds": over_odds,
                        "model_probability": round(prob_over * 100, 1),
                        "implied_probability": round(implied_over * 100, 1),
                        "edge": round((prob_over - implied_over) * 100, 1),
                        "ev": round(ev_over * 100, 1),
                        "model_prediction": pred.get("prediction")
                    })

                if ev_under >= min_ev:
                    value_bets.append({
                        "player": player,
                        "opponent": prop["opponent"],
                        "stat": stat,
                        "bet_type": "UNDER",
                        "line": line,
                        "odds": under_odds,
                        "model_probability": round(prob_under * 100, 1),
                        "implied_probability": round(implied_under * 100, 1),
                        "edge": round((prob_under - implied_under) * 100, 1),
                        "ev": round(ev_under * 100, 1),
                        "model_prediction": pred.get("prediction")
                    })

        # Ordenar por EV descendente
        value_bets.sort(key=lambda x: x["ev"], reverse=True)

        return value_bets


def scan_for_value():
    """
    Escanea el mercado buscando apuestas con valor.
    Combina cuotas del mercado con predicciones del modelo.
    """
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent / "models"))

    print("="*60)
    print("ESCÁNER DE VALOR - NBA PROPS")
    print("="*60)

    # Obtener cuotas
    fetcher = OddsFetcher()
    props = fetcher.get_player_props()

    print(f"\nProps encontrados: {len(props)} jugadores")

    # Obtener predicciones
    try:
        from xgboost_predictor import NBAPropsPredictor
        predictor = NBAPropsPredictor()

        predictions = {}

        for prop in props:
            player = prop["player"]
            opponent = prop["opponent"]

            print(f"\nAnalizando: {player} vs {opponent}")

            for stat in ["pts", "reb", "ast"]:
                try:
                    result = predictor.predict_player(player, opponent)

                    if "error" not in result:
                        pred_value = result["predictions"][stat]
                        std_value = result["std"][stat]

                        # Calcular probabilidad de over para cada línea
                        line = prop["props"][stat]["line"]
                        prob_over = predictor.calculate_over_probability(
                            pred_value, line, std_value
                        )

                        if player not in predictions:
                            predictions[player] = {}

                        predictions[player][stat] = {
                            "prediction": pred_value,
                            "prob_over": prob_over
                        }
                except Exception as e:
                    print(f"  Error con {player}/{stat}: {e}")

    except Exception as e:
        print(f"\nModelo no disponible: {e}")
        print("Usando predicciones mock para demo...")

        # Predicciones mock
        predictions = {
            "LeBron James": {
                "pts": {"prediction": 26.5, "prob_over": 0.58},
                "reb": {"prediction": 7.8, "prob_over": 0.55},
                "ast": {"prediction": 8.2, "prob_over": 0.45}
            },
            "Stephen Curry": {
                "pts": {"prediction": 27.0, "prob_over": 0.42},
                "reb": {"prediction": 5.2, "prob_over": 0.45},
                "ast": {"prediction": 6.1, "prob_over": 0.62}
            },
            "Jayson Tatum": {
                "pts": {"prediction": 29.0, "prob_over": 0.58},
                "reb": {"prediction": 8.8, "prob_over": 0.55},
                "ast": {"prediction": 4.8, "prob_over": 0.55}
            },
            "Giannis Antetokounmpo": {
                "pts": {"prediction": 32.5, "prob_over": 0.55},
                "reb": {"prediction": 12.0, "prob_over": 0.55},
                "ast": {"prediction": 5.8, "prob_over": 0.55}
            }
        }

    # Buscar valor
    calculator = EVCalculator()
    value_bets = calculator.find_value_bets(props, predictions, min_ev=0.03)

    print("\n" + "="*60)
    print("APUESTAS CON VALOR ENCONTRADAS")
    print("="*60)

    if not value_bets:
        print("\nNo se encontraron apuestas con EV > 3%")
    else:
        for i, bet in enumerate(value_bets[:10], 1):
            print(f"\n#{i} {bet['player']} - {bet['stat'].upper()} {bet['bet_type']}")
            print(f"   Línea: {bet['line']} @ {bet['odds']}")
            print(f"   Predicción modelo: {bet['model_prediction']}")
            print(f"   Prob modelo: {bet['model_probability']}% vs Mercado: {bet['implied_probability']}%")
            print(f"   Edge: +{bet['edge']}% | EV: +{bet['ev']}%")

    return value_bets


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Escáner de valor NBA Props")
    parser.add_argument("--scan", action="store_true", help="Escanear mercado")
    parser.add_argument("--min-ev", type=float, default=0.05, help="EV mínimo (default 5%)")

    args = parser.parse_args()

    if args.scan:
        scan_for_value()
    else:
        # Demo
        fetcher = OddsFetcher()
        props = fetcher.get_player_props()

        print("Props disponibles:")
        for prop in props:
            print(f"\n{prop['player']} vs {prop['opponent']}")
            for stat, data in prop['props'].items():
                print(f"  {stat.upper()}: {data['line']} (O: {data['over_odds']} / U: {data['under_odds']})")
