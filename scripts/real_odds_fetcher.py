"""
Fetcher de odds reales desde fuentes públicas accesibles.
Usa múltiples fuentes y fallbacks.
"""

import httpx
import json
import re
from datetime import datetime
from bs4 import BeautifulSoup
from typing import Optional, Dict, List


class RealOddsFetcher:
    """Obtiene odds reales de fuentes públicas."""

    def __init__(self):
        self.client = httpx.Client(
            timeout=30.0,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }
        )
        self.props_cache = {}

    def fetch_prizepicks_lines(self) -> List[Dict]:
        """
        PrizePicks muestra líneas públicamente.
        Sus líneas son un buen proxy de las líneas de mercado.
        """
        try:
            # PrizePicks API pública
            url = "https://api.prizepicks.com/projections?league_id=7"  # 7 = NBA

            print(f"    Intentando: {url}")
            response = self.client.get(url, headers={
                "Accept": "application/json",
            })
            print(f"    Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()

                # Debug: guardar respuesta
                with open("prizepicks_debug.json", "w") as f:
                    json.dump(data, f, indent=2)
                print(f"    Respuesta guardada en prizepicks_debug.json")
                print(f"    Keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")

                props = []

                # PrizePicks estructura: data.included tiene jugadores, data.data tiene proyecciones
                players = {}
                included = data.get("included", [])
                print(f"    Included items: {len(included)}")

                for item in included:
                    if item.get("type") == "new_player":
                        players[item["id"]] = item.get("attributes", {}).get("name", "")

                print(f"    Players encontrados: {len(players)}")

                projections = data.get("data", [])
                print(f"    Proyecciones: {len(projections)}")

                for proj in projections:
                    attrs = proj.get("attributes", {})
                    player_id = proj.get("relationships", {}).get("new_player", {}).get("data", {}).get("id")
                    player_name = players.get(player_id, "Unknown")

                    stat_type = attrs.get("stat_type", "")
                    line = attrs.get("line_score", 0)

                    if player_name and line and stat_type:
                        props.append({
                            "player": player_name,
                            "stat": self._normalize_stat(stat_type),
                            "line": float(line),
                            "source": "PrizePicks",
                            "odds_over": 1.91,  # PrizePicks no tiene odds tradicionales
                            "odds_under": 1.91,
                        })

                print(f"  ✓ PrizePicks: {len(props)} props encontrados")
                return props
            else:
                print(f"    Response: {response.text[:300]}")

        except Exception as e:
            print(f"  ✗ PrizePicks error: {e}")
            import traceback
            traceback.print_exc()

        return []

    def fetch_underdog_lines(self) -> List[Dict]:
        """
        Underdog Fantasy también muestra líneas públicas.
        """
        try:
            url = "https://api.underdogfantasy.com/beta/v5/over_under_lines"

            response = self.client.get(url, headers={
                "Accept": "application/json",
            })

            if response.status_code == 200:
                data = response.json()
                props = []

                for line in data.get("over_under_lines", []):
                    player_name = line.get("over_under", {}).get("appearance", {}).get("player", {}).get("full_name", "")
                    stat = line.get("over_under", {}).get("stat", "")
                    value = line.get("stat_value", 0)

                    if player_name and stat and value:
                        props.append({
                            "player": player_name,
                            "stat": self._normalize_stat(stat),
                            "line": float(value),
                            "source": "Underdog",
                            "odds_over": 1.91,
                            "odds_under": 1.91,
                        })

                print(f"  ✓ Underdog: {len(props)} props encontrados")
                return props

        except Exception as e:
            print(f"  ✗ Underdog error: {e}")

        return []

    def fetch_actionnetwork_odds(self) -> List[Dict]:
        """
        Action Network a veces tiene datos accesibles.
        """
        try:
            url = "https://www.actionnetwork.com/nba/props"
            response = self.client.get(url)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                # Buscar datos JSON embebidos
                scripts = soup.find_all('script', type='application/json')
                for script in scripts:
                    try:
                        data = json.loads(script.string)
                        # Parsear estructura de Action Network
                        if "props" in str(data).lower():
                            print(f"  ✓ Action Network: Datos encontrados")
                            return self._parse_action_network(data)
                    except:
                        continue

        except Exception as e:
            print(f"  ✗ Action Network error: {e}")

        return []

    def fetch_covers_props(self) -> List[Dict]:
        """
        Covers.com tiene props con odds reales.
        """
        try:
            url = "https://www.covers.com/sport/basketball/nba/props"
            response = self.client.get(url)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                props = []

                # Buscar tablas de props
                prop_rows = soup.find_all('tr', class_=re.compile(r'prop'))

                for row in prop_rows:
                    try:
                        cells = row.find_all('td')
                        if len(cells) >= 4:
                            player = cells[0].get_text(strip=True)
                            prop_type = cells[1].get_text(strip=True)
                            line = cells[2].get_text(strip=True)
                            odds = cells[3].get_text(strip=True)

                            if player and line:
                                props.append({
                                    "player": player,
                                    "stat": self._normalize_stat(prop_type),
                                    "line": self._parse_line(line),
                                    "source": "Covers",
                                    "odds_over": self._parse_odds(odds),
                                    "odds_under": 1.91,
                                })
                    except:
                        continue

                if props:
                    print(f"  ✓ Covers: {len(props)} props encontrados")
                return props

        except Exception as e:
            print(f"  ✗ Covers error: {e}")

        return []

    def fetch_oddschecker(self) -> List[Dict]:
        """
        OddsChecker agrega odds de múltiples books.
        """
        try:
            url = "https://www.oddschecker.com/us/basketball/nba/player-props"
            response = self.client.get(url)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                # Buscar JSON data
                script = soup.find('script', id='__NEXT_DATA__')
                if script:
                    data = json.loads(script.string)
                    # Parsear estructura
                    print(f"  ✓ OddsChecker: Datos encontrados")
                    return self._parse_oddschecker(data)

        except Exception as e:
            print(f"  ✗ OddsChecker error: {e}")

        return []

    def fetch_all_sources(self) -> List[Dict]:
        """
        Intenta todas las fuentes y combina resultados.
        """
        print("\n  Buscando odds reales...")

        all_props = []

        # Intentar cada fuente
        sources = [
            ("PrizePicks", self.fetch_prizepicks_lines),
            ("Underdog", self.fetch_underdog_lines),
            ("Covers", self.fetch_covers_props),
        ]

        for name, fetcher in sources:
            try:
                props = fetcher()
                all_props.extend(props)
            except:
                continue

        # Deduplicar por jugador+stat, manteniendo el primero encontrado
        seen = set()
        unique_props = []
        for prop in all_props:
            key = (prop["player"].lower(), prop["stat"])
            if key not in seen:
                seen.add(key)
                unique_props.append(prop)

        print(f"\n  Total props únicos: {len(unique_props)}")
        return unique_props

    def get_player_prop(self, player_name: str, stat: str = "pts") -> Optional[Dict]:
        """
        Busca la línea de un jugador específico.
        """
        if not self.props_cache:
            self.props_cache = {
                (p["player"].lower(), p["stat"]): p
                for p in self.fetch_all_sources()
            }

        key = (player_name.lower(), stat)
        return self.props_cache.get(key)

    def _normalize_stat(self, stat: str) -> str:
        """Normaliza nombres de stats."""
        stat = stat.lower()
        if "point" in stat or stat == "pts":
            return "pts"
        elif "rebound" in stat or stat == "reb":
            return "reb"
        elif "assist" in stat or stat == "ast":
            return "ast"
        elif "three" in stat or "3" in stat:
            return "fg3m"
        elif "steal" in stat:
            return "stl"
        elif "block" in stat:
            return "blk"
        return stat

    def _parse_line(self, line_str: str) -> float:
        """Parsea línea de string a float."""
        try:
            # Remover texto, quedarse solo con número
            num = re.search(r'[\d.]+', line_str)
            if num:
                return float(num.group())
        except:
            pass
        return 0.0

    def _parse_odds(self, odds_str: str) -> float:
        """Parsea odds a formato decimal."""
        try:
            odds_str = odds_str.strip()

            # American odds (+150, -110)
            if odds_str.startswith('+'):
                return 1 + (int(odds_str[1:]) / 100)
            elif odds_str.startswith('-'):
                return 1 + (100 / abs(int(odds_str)))
            else:
                # Ya es decimal
                return float(odds_str)
        except:
            return 1.91

    def _parse_action_network(self, data: dict) -> List[Dict]:
        """Parsea datos de Action Network."""
        # Implementar según estructura real
        return []

    def _parse_oddschecker(self, data: dict) -> List[Dict]:
        """Parsea datos de OddsChecker."""
        # Implementar según estructura real
        return []


def try_ballDontLie_api() -> List[Dict]:
    """
    Ball Don't Lie es una API gratuita de stats de NBA.
    No tiene odds pero tiene datos de juegos de hoy.
    """
    try:
        client = httpx.Client(timeout=30.0)
        today = datetime.now().strftime("%Y-%m-%d")
        url = f"https://api.balldontlie.io/v1/games?dates[]={today}"

        response = client.get(url, headers={
            "Authorization": "YOUR_API_KEY"  # Free tier available
        })

        print(f"  BallDontLie status: {response.status_code}")
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"  BallDontLie error: {e}")
    return []


def try_odds_api_free() -> List[Dict]:
    """
    The Odds API tiene un tier gratis (500 req/mes).
    Solo necesitas registrarte en https://the-odds-api.com
    """
    import os
    api_key = os.environ.get("ODDS_API_KEY", "")

    if not api_key:
        print("  ⚠️ ODDS_API_KEY no configurada")
        print("  → Regístrate gratis en https://the-odds-api.com")
        print("  → 500 requests/mes gratis (suficiente para uso diario)")
        return []

    client = httpx.Client(timeout=30.0)

    # Primero: obtener eventos de NBA
    print("  Obteniendo eventos NBA...")
    events_url = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
    response = client.get(events_url, params={"apiKey": api_key})
    print(f"    Events status: {response.status_code}")

    if response.status_code == 200:
        events = response.json()
        print(f"    Eventos hoy: {len(events)}")

        if events:
            # Mostrar primer evento
            print(f"    Ejemplo: {events[0].get('away_team')} @ {events[0].get('home_team')}")

            # Intentar obtener player props para el primer evento
            event_id = events[0].get("id")
            print(f"\n  Obteniendo player props para evento {event_id}...")

            props_url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{event_id}/odds"
            props_response = client.get(props_url, params={
                "apiKey": api_key,
                "regions": "us",
                "markets": "player_points",
                "oddsFormat": "decimal"
            })
            print(f"    Props status: {props_response.status_code}")

            if props_response.status_code == 200:
                props_data = props_response.json()
                # Guardar para debug
                with open("odds_api_debug.json", "w") as f:
                    json.dump(props_data, f, indent=2)
                print(f"    Guardado en odds_api_debug.json")
                return props_data
            else:
                print(f"    Error: {props_response.text[:200]}")

                # Si player props no funciona, intentar solo odds de juegos
                print("\n  Intentando odds de juegos (spread/totals)...")
                games_url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
                games_response = client.get(games_url, params={
                    "apiKey": api_key,
                    "regions": "us",
                    "markets": "h2h,spreads,totals",
                    "oddsFormat": "decimal"
                })
                print(f"    Games odds status: {games_response.status_code}")

                if games_response.status_code == 200:
                    games_data = games_response.json()
                    print(f"    ✓ Juegos con odds: {len(games_data)}")
                    with open("odds_api_games.json", "w") as f:
                        json.dump(games_data, f, indent=2)
                    return games_data

    else:
        print(f"    Error: {response.text[:200]}")

    return []


def try_sportsdata_io() -> List[Dict]:
    """
    SportsData.io tiene tier gratis para desarrollo.
    """
    import os
    api_key = os.environ.get("SPORTSDATA_API_KEY", "")

    if not api_key:
        print("  ⚠️ SPORTSDATA_API_KEY no configurada (opcional)")
        return []

    try:
        client = httpx.Client(timeout=30.0)
        url = f"https://api.sportsdata.io/v3/nba/odds/json/BettingPlayerPropsByDate/{datetime.now().strftime('%Y-%m-%d')}"

        response = client.get(url, headers={
            "Ocp-Apim-Subscription-Key": api_key
        })

        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"  SportsData error: {e}")
    return []


def test_fetcher():
    """Prueba el fetcher de odds."""
    print("="*60)
    print("TESTING REAL ODDS FETCHER")
    print("="*60)

    # Primero probar APIs que requieren key (recomendado)
    print("\n[1] APIs con key gratuita:")
    odds_data = try_odds_api_free()
    if odds_data:
        print(f"  ✓ The Odds API: {len(odds_data)} eventos")

    # Luego probar scraping (menos confiable)
    print("\n[2] Scraping directo:")
    fetcher = RealOddsFetcher()
    props = fetcher.fetch_all_sources()

    if props:
        print("\n" + "="*60)
        print("PROPS ENCONTRADOS")
        print("="*60)

        # Agrupar por stat
        by_stat = {}
        for p in props:
            stat = p["stat"]
            if stat not in by_stat:
                by_stat[stat] = []
            by_stat[stat].append(p)

        for stat, stat_props in by_stat.items():
            print(f"\n{stat.upper()} ({len(stat_props)} props):")
            for p in stat_props[:5]:  # Mostrar primeros 5
                print(f"  {p['player']}: {p['line']} ({p['source']})")
    else:
        print("\n  No se encontraron props :(")
        print("  Las APIs pueden estar bloqueadas o la estructura cambió")


if __name__ == "__main__":
    test_fetcher()
