"""
Scraper de líneas reales de DraftKings.
"""

import httpx
import json
import re
from datetime import datetime
from typing import Optional


class DraftKingsScraper:
    """Scraper para obtener player props de DraftKings."""

    def __init__(self):
        self.client = httpx.Client(
            timeout=30.0,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Origin": "https://sportsbook.draftkings.com",
                "Referer": "https://sportsbook.draftkings.com/",
            }
        )
        self.base_url = "https://sportsbook.draftkings.com"

    def get_nba_events(self) -> list:
        """Obtiene los eventos/partidos de NBA disponibles."""
        # DraftKings usa una API interna para los eventos
        # El endpoint puede cambiar, pero este es el patrón común
        urls_to_try = [
            "https://sportsbook-nash.draftkings.com/api/sportscontent/dkusnj/v1/leagues/42648",  # NBA
            "https://sportsbook-nash.draftkings.com/sites/US-NJ-SB/api/v5/eventgroups/42648",
            "https://sportsbook.draftkings.com/sites/US-SB/api/v5/eventgroups/42648",
        ]

        for url in urls_to_try:
            try:
                response = self.client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    print(f"  ✓ Conectado a DraftKings API")
                    return data
            except Exception as e:
                continue

        return None

    def get_player_props_api(self) -> list:
        """
        Intenta obtener player props desde la API de DraftKings.
        """
        # Intentar diferentes endpoints de la API
        api_urls = [
            # API de ofertas de jugadores
            "https://sportsbook-nash.draftkings.com/api/sportscontent/dkusnj/v1/leagues/42648/categories/1215",  # Player Points
            "https://sportsbook-nash.draftkings.com/api/sportscontent/dkusnj/v1/leagues/42648/categories/1216",  # Player Rebounds
            "https://sportsbook-nash.draftkings.com/api/sportscontent/dkusnj/v1/leagues/42648/categories/1217",  # Player Assists
        ]

        all_props = []

        for url in api_urls:
            try:
                response = self.client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    all_props.append(data)
                    print(f"  ✓ Props encontrados en: {url.split('/')[-1]}")
            except:
                continue

        return all_props

    def get_props_from_eventgroup(self) -> list:
        """
        Obtiene props desde el endpoint de eventgroup.
        DraftKings estructura: eventgroup -> events -> offers
        """
        try:
            # NBA event group ID es típicamente 42648
            url = "https://sportsbook-nash.draftkings.com/sites/US-NJ-SB/api/v5/eventgroups/42648/categories/1215"

            response = self.client.get(url)
            print(f"  Status: {response.status_code}")

            if response.status_code == 200:
                return response.json()
            else:
                print(f"  Response: {response.text[:500]}")

        except Exception as e:
            print(f"  Error: {e}")

        return None

    def parse_props_response(self, data: dict) -> list:
        """Parsea la respuesta de props en formato estándar."""
        props = []

        if not data:
            return props

        try:
            # La estructura de DK varía, intentar diferentes paths
            events = data.get("events", data.get("eventGroup", {}).get("events", []))

            for event in events:
                offers = event.get("offers", event.get("displayGroups", []))

                for offer in offers:
                    outcomes = offer.get("outcomes", [])

                    for outcome in outcomes:
                        player_name = outcome.get("participant", outcome.get("label", ""))
                        line = outcome.get("line", outcome.get("handicap", 0))
                        odds = outcome.get("oddsDecimal", outcome.get("odds", 1.91))

                        if player_name and line:
                            props.append({
                                "player": player_name,
                                "line": float(line),
                                "odds": float(odds) if odds else 1.91,
                                "type": offer.get("label", "unknown")
                            })

        except Exception as e:
            print(f"  Error parsing: {e}")

        return props


class FanDuelScraper:
    """Scraper para FanDuel."""

    def __init__(self):
        self.client = httpx.Client(
            timeout=30.0,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json",
            }
        )

    def get_nba_props(self) -> list:
        """Intenta obtener props de FanDuel."""
        urls = [
            "https://sportsbook.fanduel.com/cache/psmg/UK/69589.3.json",  # NBA
            "https://sbapi.mi.sportsbook.fanduel.com/api/content-managed-page?page=SPORT&eventId=nba",
        ]

        for url in urls:
            try:
                response = self.client.get(url)
                if response.status_code == 200:
                    print(f"  ✓ Conectado a FanDuel")
                    return response.json()
            except:
                continue

        return None


class BetMGMScraper:
    """Scraper para BetMGM."""

    def __init__(self):
        self.client = httpx.Client(
            timeout=30.0,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            }
        )

    def get_nba_props(self) -> list:
        """Intenta obtener props de BetMGM."""
        try:
            # BetMGM API endpoint
            url = "https://sports.mi.betmgm.com/cds-api/bettingoffer/fixtures?x-bwin-accessid=NGQ0NDRjMzItMWMzNS00NTFhLWEyNDgtYjRhNmEyYWQxZTFi&lang=en-us&country=US&fixtureTypes=Standard&state=Started&offerMapping=Ede&sportIds=7"

            response = self.client.get(url)
            if response.status_code == 200:
                print(f"  ✓ Conectado a BetMGM")
                return response.json()
        except Exception as e:
            print(f"  BetMGM error: {e}")

        return None


class OddsJamScraper:
    """
    Scraper para OddsJam - agregador de odds que muestra múltiples books.
    """

    def __init__(self):
        self.client = httpx.Client(
            timeout=30.0,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
        )

    def get_nba_props(self) -> Optional[str]:
        """Obtiene página de props de OddsJam."""
        try:
            url = "https://oddsjam.com/nba/player-props"
            response = self.client.get(url)

            if response.status_code == 200:
                print(f"  ✓ Página de OddsJam obtenida ({len(response.text)} bytes)")
                return response.text
        except Exception as e:
            print(f"  OddsJam error: {e}")

        return None


def test_all_scrapers():
    """Prueba todos los scrapers disponibles."""
    print("="*60)
    print("TESTING SCRAPERS DE ODDS REALES")
    print("="*60)

    # Test DraftKings
    print("\n[1] DraftKings:")
    dk = DraftKingsScraper()
    dk_events = dk.get_nba_events()
    if dk_events:
        print(f"    Datos obtenidos: {type(dk_events)}")
        if isinstance(dk_events, dict):
            print(f"    Keys: {list(dk_events.keys())[:5]}")

    dk_props = dk.get_props_from_eventgroup()
    if dk_props:
        print(f"    Props data: {type(dk_props)}")

    # Test FanDuel
    print("\n[2] FanDuel:")
    fd = FanDuelScraper()
    fd_data = fd.get_nba_props()
    if fd_data:
        print(f"    Datos obtenidos: {type(fd_data)}")
        if isinstance(fd_data, dict):
            print(f"    Keys: {list(fd_data.keys())[:10]}")
            # Guardar para análisis
            with open("fanduel_response.json", "w") as f:
                json.dump(fd_data, f, indent=2)
            print(f"    Guardado en fanduel_response.json")

    # Test BetMGM
    print("\n[3] BetMGM:")
    mgm = BetMGMScraper()
    mgm_data = mgm.get_nba_props()
    if mgm_data:
        print(f"    Datos obtenidos: {type(mgm_data)}")

    # Test OddsJam
    print("\n[4] OddsJam:")
    oj = OddsJamScraper()
    oj_html = oj.get_nba_props()
    if oj_html:
        # Buscar datos JSON embebidos
        json_match = re.search(r'__NEXT_DATA__.*?>(.*?)</script>', oj_html)
        if json_match:
            print(f"    Found embedded JSON data")

    print("\n" + "="*60)


if __name__ == "__main__":
    test_all_scrapers()
