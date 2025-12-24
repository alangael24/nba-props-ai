"""
Fase 4: Cazador de Valor
Obtiene cuotas de apuestas y calcula Expected Value.

Fuentes de cuotas:
- The Odds API (gratuita con límites)
- Scraping directo (con cuidado)
"""

import os
import sys
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import httpx
import pandas as pd

# Agregar paths
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))
sys.path.insert(0, str(Path(__file__).parent))

# Name matcher para resolver nombres de jugadores
try:
    from utils.name_matcher import get_matcher, resolve_player_name
    NAME_MATCHER_AVAILABLE = True
except ImportError:
    NAME_MATCHER_AVAILABLE = False
    print("Warning: name_matcher not available. Using exact matching.")


# nba_api para partidos del día
try:
    from nba_api.stats.endpoints import ScoreboardV2
    from nba_api.stats.static import teams
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False

# API Key para The Odds API (gratuita: 500 requests/mes)
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Cache global de spreads por partido
GAME_SPREADS_CACHE: dict = {}  # (home_team, away_team) -> spread


def fetch_game_spreads() -> dict:
    """
    Obtiene spreads de partidos de The Odds API.

    Returns:
        Dict mapeo de (home_team, away_team) -> spread (positivo = home favored)
    """
    global GAME_SPREADS_CACHE

    if GAME_SPREADS_CACHE:
        return GAME_SPREADS_CACHE

    if not ODDS_API_KEY:
        print("  ⚠️ ODDS_API_KEY no configurada para spreads")
        return {}

    client = httpx.Client(timeout=30.0)
    spreads = {}

    try:
        # Obtener odds con spreads
        odds_url = f"{ODDS_API_BASE}/sports/basketball_nba/odds"
        response = client.get(odds_url, params={
            "apiKey": ODDS_API_KEY,
            "regions": "us",
            "markets": "spreads",
            "oddsFormat": "decimal"
        })

        if response.status_code != 200:
            print(f"  Error obteniendo spreads: {response.status_code}")
            return {}

        events = response.json()
        print(f"  📊 Spreads para {len(events)} partidos")

        for event in events:
            home_team = event.get("home_team", "")
            away_team = event.get("away_team", "")
            bookmakers = event.get("bookmakers", [])

            # Tomar el primer bookmaker con spreads
            for book in bookmakers:
                markets = book.get("markets", [])
                for market in markets:
                    if market.get("key") == "spreads":
                        outcomes = market.get("outcomes", [])
                        for outcome in outcomes:
                            team = outcome.get("name", "")
                            spread = outcome.get("point", 0)

                            # El spread es desde la perspectiva del equipo
                            # Si home tiene -5, home is favored by 5
                            if team == home_team:
                                spreads[(home_team, away_team)] = spread
                                break
                        break
                if (home_team, away_team) in spreads:
                    break

    except Exception as e:
        print(f"  Error fetching spreads: {e}")
    finally:
        client.close()

    GAME_SPREADS_CACHE = spreads
    return spreads


def get_game_spread(home_team: str, away_team: str) -> float:
    """
    Obtiene el spread para un partido específico.

    Args:
        home_team: Nombre completo del equipo local
        away_team: Nombre completo del equipo visitante

    Returns:
        Spread (negativo = home favored, positivo = away favored)
        0.0 si no se encuentra
    """
    spreads = fetch_game_spreads()
    return spreads.get((home_team, away_team), 0.0)


def estimate_blowout_risk(spread: float, is_favored: bool) -> float:
    """
    Estima el riesgo de blowout basado en el spread.

    Args:
        spread: Spread del partido (ej: -10 si el equipo está muy favorito)
        is_favored: True si el jugador está en el equipo favorito

    Returns:
        Factor de ajuste de minutos (0.8-1.0)
        - 1.0 = sin riesgo de blowout
        - 0.8 = alto riesgo de minutos reducidos
    """
    abs_spread = abs(spread)

    if abs_spread < 5:
        # Partido cerrado, sin riesgo
        return 1.0
    elif abs_spread < 8:
        # Spread moderado
        if is_favored:
            return 0.95  # Ligero riesgo
        else:
            return 0.92  # El underdog puede tener menos minutos si pierde mucho
    elif abs_spread < 12:
        # Spread grande
        if is_favored:
            return 0.90  # Riesgo moderado de blowout
        else:
            return 0.88
    else:
        # Spread muy grande (>12)
        if is_favored:
            return 0.85  # Alto riesgo de blowout
        else:
            return 0.82

    return 1.0


def fetch_real_odds_from_api() -> list:
    """
    Obtiene odds REALES de The Odds API.
    Requiere ODDS_API_KEY configurada.
    Incluye spread para calcular riesgo de blowout.
    """
    if not ODDS_API_KEY:
        print("  ⚠️ ODDS_API_KEY no configurada")
        return []

    # Pre-fetch spreads para todos los partidos
    spreads = fetch_game_spreads()

    client = httpx.Client(timeout=30.0)
    all_props = []

    try:
        # 1. Obtener todos los eventos de NBA
        events_url = f"{ODDS_API_BASE}/sports/basketball_nba/events"
        response = client.get(events_url, params={"apiKey": ODDS_API_KEY})

        if response.status_code != 200:
            print(f"  Error obteniendo eventos: {response.status_code}")
            return []

        events = response.json()
        print(f"  📅 Eventos NBA: {len(events)}")

        # 2. Obtener player props para cada evento
        for event in events:
            event_id = event.get("id")
            home_team = event.get("home_team", "")
            away_team = event.get("away_team", "")

            # Mapear nombres de equipo a abreviaturas
            team_abbrevs = {
                "Philadelphia 76ers": "PHI", "Brooklyn Nets": "BKN",
                "Los Angeles Lakers": "LAL", "Golden State Warriors": "GSW",
                "Boston Celtics": "BOS", "Milwaukee Bucks": "MIL",
                "Phoenix Suns": "PHX", "Denver Nuggets": "DEN",
                "Miami Heat": "MIA", "Cleveland Cavaliers": "CLE",
                "New York Knicks": "NYK", "Chicago Bulls": "CHI",
                "Dallas Mavericks": "DAL", "Memphis Grizzlies": "MEM",
                "Atlanta Hawks": "ATL", "Toronto Raptors": "TOR",
                "Oklahoma City Thunder": "OKC", "Minnesota Timberwolves": "MIN",
                "Sacramento Kings": "SAC", "Indiana Pacers": "IND",
                "New Orleans Pelicans": "NOP", "Orlando Magic": "ORL",
                "Charlotte Hornets": "CHA", "Houston Rockets": "HOU",
                "San Antonio Spurs": "SAS", "Portland Trail Blazers": "POR",
                "Utah Jazz": "UTA", "Washington Wizards": "WAS",
                "Detroit Pistons": "DET", "LA Clippers": "LAC"
            }

            home_abbrev = team_abbrevs.get(home_team, home_team[:3].upper())
            away_abbrev = team_abbrevs.get(away_team, away_team[:3].upper())

            # Obtener props para este evento
            props_url = f"{ODDS_API_BASE}/sports/basketball_nba/events/{event_id}/odds"

            for market in ["player_points", "player_rebounds", "player_assists"]:
                try:
                    props_response = client.get(props_url, params={
                        "apiKey": ODDS_API_KEY,
                        "regions": "us",
                        "markets": market,
                        "oddsFormat": "decimal"
                    })

                    if props_response.status_code == 200:
                        data = props_response.json()
                        bookmakers = data.get("bookmakers", [])

                        for book in bookmakers:
                            book_name = book.get("title", "Unknown")
                            markets = book.get("markets", [])

                            for mkt in markets:
                                outcomes = mkt.get("outcomes", [])

                                # Agrupar por jugador
                                player_lines = {}
                                for outcome in outcomes:
                                    player = outcome.get("description", "")
                                    line = outcome.get("point", 0)
                                    odds = outcome.get("price", 1.91)
                                    over_under = outcome.get("name", "")

                                    if player not in player_lines:
                                        player_lines[player] = {"lines": {}}

                                    if line not in player_lines[player]["lines"]:
                                        player_lines[player]["lines"][line] = {}

                                    if over_under == "Over":
                                        player_lines[player]["lines"][line]["over"] = odds
                                    else:
                                        player_lines[player]["lines"][line]["under"] = odds

                                # Agregar props
                                stat = market.replace("player_", "")[:3]  # points -> pts
                                for player, data in player_lines.items():
                                    # Usar la línea más común (típicamente la del medio)
                                    lines = list(data["lines"].keys())
                                    if lines:
                                        # Tomar la línea con odds más cercanos a 1.91
                                        best_line = None
                                        best_balance = float('inf')
                                        for line_val in lines:
                                            line_data = data["lines"][line_val]
                                            over = line_data.get("over", 1.91)
                                            under = line_data.get("under", 1.91)
                                            balance = abs(over - under)
                                            if balance < best_balance:
                                                best_balance = balance
                                                best_line = line_val

                                        if best_line:
                                            line_data = data["lines"][best_line]

                                            # Obtener spread del partido
                                            game_spread = spreads.get((home_team, away_team), 0.0)

                                            all_props.append({
                                                "player": player,
                                                "team": home_abbrev if player else away_abbrev,
                                                "opponent": away_abbrev,
                                                "home_team": home_team,
                                                "away_team": away_team,
                                                "stat": stat,
                                                "line": best_line,
                                                "over_odds": line_data.get("over", 1.91),
                                                "under_odds": line_data.get("under", 1.91),
                                                "source": book_name,
                                                "game_spread": game_spread
                                            })

                except Exception as e:
                    continue

            # Pequeña pausa para no saturar la API
            import time
            time.sleep(0.1)

    except Exception as e:
        print(f"  Error: {e}")

    # Deduplicar: quedarse con el primer registro por jugador+stat
    seen = set()
    unique_props = []
    for prop in all_props:
        key = (prop["player"], prop["stat"])
        if key not in seen:
            seen.add(key)
            unique_props.append(prop)

    print(f"  ✅ Props únicos: {len(unique_props)}")
    return unique_props


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
        # Intentar obtener partidos reales de hoy
        if NBA_API_AVAILABLE:
            try:
                return self._get_todays_real_games()
            except Exception as e:
                print(f"  (Usando partidos mock: {e})")

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

    def _get_todays_real_games(self) -> list:
        """Obtiene los partidos reales programados para hoy desde NBA API."""
        import time
        from nba_api.stats.endpoints import ScoreboardV2

        # Obtener scoreboard de hoy
        scoreboard = ScoreboardV2(game_date=datetime.now().strftime("%Y-%m-%d"))
        time.sleep(1)

        games_df = scoreboard.get_data_frames()[0]

        if games_df.empty:
            print("  No hay partidos programados para hoy")
            return []

        real_games = []
        team_dict = {t["id"]: t for t in teams.get_teams()}

        for _, game in games_df.iterrows():
            home_id = game.get("HOME_TEAM_ID")
            away_id = game.get("VISITOR_TEAM_ID")

            home_team = team_dict.get(home_id, {}).get("full_name", "Unknown")
            away_team = team_dict.get(away_id, {}).get("full_name", "Unknown")

            real_games.append({
                "id": str(game.get("GAME_ID", "")),
                "sport_key": "basketball_nba",
                "home_team": home_team,
                "away_team": away_team,
                "home_abbrev": team_dict.get(home_id, {}).get("abbreviation", ""),
                "away_abbrev": team_dict.get(away_id, {}).get("abbreviation", ""),
                "commence_time": datetime.now().isoformat()
            })

        print(f"  Encontrados {len(real_games)} partidos para hoy")
        return real_games

    def _get_mock_player_props(self, player_name: str = None) -> list:
        """
        Genera props basados en partidos reales de hoy.
        Usa líneas aproximadas basadas en promedios históricos.
        """
        # Obtener partidos de hoy
        games = self._get_todays_real_games() if NBA_API_AVAILABLE else []

        if not games:
            # Si no hay partidos reales, usar mock estático
            return self._get_static_mock_props(player_name)

        # Obtener top jugadores de cada equipo desde nuestra DB
        db_path = Path(__file__).parent.parent / "data" / "nba_props.db"
        if not db_path.exists():
            return self._get_static_mock_props(player_name)

        conn = sqlite3.connect(db_path)
        props = []
        seen_players = set()  # Para evitar duplicados

        for game in games:
            home_abbrev = game.get("home_abbrev", "")
            away_abbrev = game.get("away_abbrev", "")

            if not home_abbrev or not away_abbrev:
                continue

            # Obtener top 3 jugadores de cada equipo por puntos promedio
            for team_abbrev, opponent_abbrev, is_home in [
                (home_abbrev, away_abbrev, True),
                (away_abbrev, home_abbrev, False)
            ]:
                try:
                    # Buscar jugadores del equipo - matchup empieza con la abreviatura del equipo
                    query = """
                        SELECT player_name,
                               AVG(pts) as avg_pts,
                               AVG(reb) as avg_reb,
                               AVG(ast) as avg_ast,
                               COUNT(*) as games
                        FROM player_game_logs
                        WHERE (matchup LIKE ? OR matchup LIKE ?)
                          AND season = '2024-25'
                        GROUP BY player_name
                        HAVING games >= 5
                        ORDER BY avg_pts DESC
                        LIMIT 3
                    """
                    # Matchup format: "LAL vs. GSW" o "LAL @ GSW"
                    df = pd.read_sql(query, conn, params=[
                        f"{team_abbrev} vs.%",
                        f"{team_abbrev} @%"
                    ])

                    for _, player in df.iterrows():
                        player_name_db = player["player_name"]

                        # Evitar duplicados
                        if player_name_db in seen_players:
                            continue
                        seen_players.add(player_name_db)

                        # Crear líneas basadas en promedios (redondeadas a .5)
                        pts_line = round(player["avg_pts"] * 2) / 2
                        reb_line = round(player["avg_reb"] * 2) / 2
                        ast_line = round(player["avg_ast"] * 2) / 2

                        props.append({
                            "player": player_name_db,
                            "team": team_abbrev,
                            "opponent": opponent_abbrev,
                            "is_home": is_home,
                            "props": {
                                "pts": {"line": pts_line, "over_odds": 1.91, "under_odds": 1.91},
                                "reb": {"line": reb_line, "over_odds": 1.91, "under_odds": 1.91},
                                "ast": {"line": ast_line, "over_odds": 1.91, "under_odds": 1.91}
                            }
                        })
                except Exception as e:
                    print(f"  Error buscando jugadores de {team_abbrev}: {e}")
                    continue

        conn.close()

        if not props:
            return self._get_static_mock_props(player_name)

        if player_name:
            return [p for p in props if player_name.lower() in p["player"].lower()]

        return props

    def _get_static_mock_props(self, player_name: str = None) -> list:
        """Props mock estáticos cuando no hay datos reales."""
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
                pred_value = pred.get("prediction", 0)
                pred_rounded = round(pred_value, 1) if pred_value else 0

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
                        "model_prediction": pred_rounded
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
                        "model_prediction": pred_rounded
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

    # Intentar obtener odds reales primero
    real_props = []
    if ODDS_API_KEY:
        print("\n🔴 USANDO ODDS REALES (The Odds API)")
        real_props = fetch_real_odds_from_api()

    if real_props:
        # Convertir formato de API a formato esperado
        props = []
        seen_players = set()

        for rp in real_props:
            player = rp["player"]
            game_spread = rp.get("game_spread", 0.0)
            home_team = rp.get("home_team", "")
            away_team = rp.get("away_team", "")

            if player in seen_players:
                # Ya existe, agregar stat
                for p in props:
                    if p["player"] == player:
                        p["props"][rp["stat"]] = {
                            "line": rp["line"],
                            "over_odds": rp["over_odds"],
                            "under_odds": rp["under_odds"]
                        }
                        break
            else:
                seen_players.add(player)
                props.append({
                    "player": player,
                    "opponent": rp["opponent"],
                    "is_home": True,  # TODO: determinar correctamente
                    "source": rp.get("source", "API"),
                    "game_spread": game_spread,
                    "home_team": home_team,
                    "away_team": away_team,
                    "props": {
                        rp["stat"]: {
                            "line": rp["line"],
                            "over_odds": rp["over_odds"],
                            "under_odds": rp["under_odds"]
                        }
                    }
                })

        print(f"\n✅ Props reales: {len(props)} jugadores")
    else:
        print("\n⚠️ Usando datos estimados (sin API key)")
        fetcher = OddsFetcher()
        props = fetcher.get_player_props()
        print(f"\nProps encontrados: {len(props)} jugadores")

    # Obtener predicciones
    try:
        from xgboost_predictor import NBAPropsPredictor
        predictor = NBAPropsPredictor()

        # Inicializar name matcher si está disponible
        name_matcher = get_matcher() if NAME_MATCHER_AVAILABLE else None
        resolved_count = 0
        unresolved_count = 0

        predictions = {}

        for prop in props:
            player_raw = prop["player"]
            opponent = prop["opponent"]
            is_home = prop.get("is_home", True)

            # Resolver nombre usando fuzzy matching
            if name_matcher:
                resolved = name_matcher.resolve(player_raw)
                if resolved:
                    player = resolved[0]  # Nombre canónico
                    if player != player_raw:
                        resolved_count += 1
                else:
                    player = player_raw
                    unresolved_count += 1
            else:
                player = player_raw

            # Evitar predecir el mismo jugador múltiples veces
            if player in predictions:
                continue

            source = prop.get("source", "")
            source_tag = f" [{source}]" if source else ""

            # Obtener spread y calcular riesgo de blowout
            game_spread = prop.get("game_spread", 0.0)
            is_favored = game_spread < 0 if is_home else game_spread > 0
            blowout_factor = estimate_blowout_risk(game_spread, is_favored)

            spread_info = ""
            if abs(game_spread) >= 8:
                spread_info = f" [Spread: {game_spread:+.1f}, Blowout Risk: {(1-blowout_factor)*100:.0f}%]"

            print(f"\nAnalizando: {player} vs {opponent}{source_tag}{spread_info}")

            try:
                result = predictor.predict_player(player, opponent, is_home=is_home)

                if "error" in result:
                    print(f"  ⚠️ {result['error']}")
                    continue

                predictions[player] = {}

                # Solo procesar stats que existen en props
                available_stats = list(prop.get("props", {}).keys())

                for stat in available_stats:
                    if stat not in ["pts", "reb", "ast"]:
                        continue

                    pred_value = result["predictions"].get(stat, 0)
                    stat_data = prop["props"].get(stat, {})
                    line = stat_data.get("line", 0)

                    if not line:
                        continue

                    # Usar cuantiles reales para todas las stats
                    if stat == "pts":
                        p15 = result["predictions_pts"]["p15_floor"]
                        p50 = result["predictions_pts"]["p50_median"]
                        p85 = result["predictions_pts"]["p85_ceiling"]
                    elif stat == "reb":
                        p15 = result["predictions_reb"]["p15_floor"]
                        p50 = result["predictions_reb"]["p50_median"]
                        p85 = result["predictions_reb"]["p85_ceiling"]
                    elif stat == "ast":
                        p15 = result["predictions_ast"]["p15_floor"]
                        p50 = result["predictions_ast"]["p50_median"]
                        p85 = result["predictions_ast"]["p85_ceiling"]

                    prob_over = predictor.calculate_probability_from_quantiles(
                        p15, p50, p85, line
                    )

                    predictions[player][stat] = {
                        "prediction": pred_value,
                        "prob_over": prob_over,
                        "p15": p15,
                        "p85": p85,
                        "blowout_factor": blowout_factor,
                        "game_spread": game_spread
                    }

            except Exception as e:
                print(f"  Error con {player}: {e}")

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
            pred = f"{bet['model_prediction']:.1f}" if bet['model_prediction'] else "N/A"
            prob = f"{bet['model_probability']:.1f}"
            edge = f"{bet['edge']:.1f}"
            ev = f"{bet['ev']:.1f}"

            # Verificar riesgo de blowout desde predictions
            player = bet['player']
            stat = bet['stat']
            blowout_warning = ""
            if player in predictions and stat in predictions[player]:
                blowout_factor = predictions[player][stat].get("blowout_factor", 1.0)
                game_spread = predictions[player][stat].get("game_spread", 0.0)
                if blowout_factor < 0.92:
                    blowout_warning = f" ⚠️ BLOWOUT RISK (spread: {game_spread:+.1f})"

            print(f"\n#{i} {bet['player']} - {bet['stat'].upper()} {bet['bet_type']}{blowout_warning}")
            print(f"   Línea: {bet['line']} @ {bet['odds']}")
            print(f"   Predicción: {pred} | Prob: {prob}%")
            print(f"   Edge: +{edge}% | EV: +{ev}%")

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
