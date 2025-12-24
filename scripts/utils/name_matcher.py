"""
Player Name Resolution System

Resuelve el problema de nombres inconsistentes entre fuentes:
- NBA API: "Nicolas Claxton"
- DraftKings: "Nic Claxton"
- FanDuel: "N. Claxton"

Usa fuzzy matching + mapeos manuales para casos conocidos.
"""

import sqlite3
from pathlib import Path
from typing import Optional, Dict, Tuple
import unicodedata
import re

try:
    from fuzzywuzzy import fuzz, process
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    print("Warning: fuzzywuzzy not installed. Using exact matching only.")

import pandas as pd

DB_PATH = Path(__file__).parent.parent.parent / "data" / "nba_props.db"


# Mapeos manuales para casos conocidos problemáticos
MANUAL_MAPPINGS = {
    # Sufijos Jr/Sr
    "jaren jackson jr": "Jaren Jackson Jr.",
    "jaren jackson": "Jaren Jackson Jr.",
    "wendell carter jr": "Wendell Carter Jr.",
    "wendell carter": "Wendell Carter Jr.",
    "jabari smith jr": "Jabari Smith Jr.",
    "jabari smith": "Jabari Smith Jr.",
    "marcus morris sr": "Marcus Morris Sr.",
    "gary trent jr": "Gary Trent Jr.",
    "larry nance jr": "Larry Nance Jr.",
    "tim hardaway jr": "Tim Hardaway Jr.",
    "kelly oubre jr": "Kelly Oubre Jr.",
    "derrick jones jr": "Derrick Jones Jr.",
    "kenyon martin jr": "Kenyon Martin Jr.",
    "troy brown jr": "Troy Brown Jr.",
    "dennis smith jr": "Dennis Smith Jr.",
    "michael porter jr": "Michael Porter Jr.",
    "kevin porter jr": "Kevin Porter Jr.",
    "otto porter jr": "Otto Porter Jr.",
    "gary payton ii": "Gary Payton II",

    # Caracteres especiales / acentos
    "nikola jokic": "Nikola Jokić",
    "nikola jokić": "Nikola Jokić",
    "luka doncic": "Luka Dončić",
    "luka dončić": "Luka Dončić",
    "dennis schroder": "Dennis Schröder",
    "dennis schröder": "Dennis Schröder",
    "dennis schroeder": "Dennis Schröder",
    "jonas valanciunas": "Jonas Valančiūnas",
    "jonas valančiūnas": "Jonas Valančiūnas",
    "domantas sabonis": "Domantas Sabonis",
    "kristaps porzingis": "Kristaps Porziņģis",
    "bogdan bogdanovic": "Bogdan Bogdanović",
    "bojan bogdanovic": "Bojan Bogdanović",
    "goran dragic": "Goran Dragić",
    "jusuf nurkic": "Jusuf Nurkić",
    "nikola vucevic": "Nikola Vučević",
    "nikola vučević": "Nikola Vučević",
    "vasilije micic": "Vasilije Micić",
    "alperen sengun": "Alperen Şengün",
    "cedi osman": "Cedi Osman",
    "enes kanter": "Enes Freedom",
    "enes freedom": "Enes Freedom",

    # Apodos y variaciones
    "nic claxton": "Nicolas Claxton",
    "nicolas claxton": "Nicolas Claxton",
    "herb jones": "Herbert Jones",
    "herbert jones": "Herbert Jones",
    "pj washington": "P.J. Washington",
    "p.j. washington": "P.J. Washington",
    "og anunoby": "OG Anunoby",
    "o.g. anunoby": "OG Anunoby",
    "cj mccollum": "CJ McCollum",
    "c.j. mccollum": "CJ McCollum",
    "rj barrett": "RJ Barrett",
    "r.j. barrett": "RJ Barrett",
    "tj mcconnell": "T.J. McConnell",
    "tj warren": "T.J. Warren",
    "jt thor": "JT Thor",
    "gg jackson": "GG Jackson II",
    "g.g. jackson": "GG Jackson II",
    "a.j. green": "A.J. Green",
    "aj green": "A.J. Green",

    # Nombres compuestos
    "shai gilgeous-alexander": "Shai Gilgeous-Alexander",
    "shai gilgeous alexander": "Shai Gilgeous-Alexander",
    "kentavious caldwell-pope": "Kentavious Caldwell-Pope",
    "kentavious caldwell pope": "Kentavious Caldwell-Pope",
    "karl-anthony towns": "Karl-Anthony Towns",
    "karl anthony towns": "Karl-Anthony Towns",
    "de'aaron fox": "De'Aaron Fox",
    "deaaron fox": "De'Aaron Fox",
    "de'anthony melton": "De'Anthony Melton",
    "deanthony melton": "De'Anthony Melton",

    # Otros casos
    "naz reid": "Naz Reid",
    "nazreon reid": "Naz Reid",
    "moe wagner": "Moritz Wagner",
    "moritz wagner": "Moritz Wagner",
    "ish smith": "Ishmael Smith",
    "ishmael smith": "Ishmael Smith",

    # Isaiah vs Isiah
    "isaiah stewart ii": "Isaiah Stewart",
    "isaiah stewart": "Isaiah Stewart",

    # Jimmy Butler especial
    "jimmy butler": "Jimmy Butler",
    "jimmy butler iii": "Jimmy Butler",

    # Más variaciones comunes de PrizePicks/DraftKings
    "cam thomas": "Cameron Thomas",
    "cameron thomas": "Cameron Thomas",
    "cam johnson": "Cameron Johnson",
    "cameron johnson": "Cameron Johnson",
    "alex caruso": "Alex Caruso",
    "tre jones": "Tre Jones",
    "trey murphy iii": "Trey Murphy III",
    "trey murphy": "Trey Murphy III",
    "jalen williams": "Jalen Williams",
    "j williams": "Jalen Williams",
    "jonathan isaac": "Jonathan Isaac",
    "mo bamba": "Mohamed Bamba",
    "mohamed bamba": "Mohamed Bamba",
    "scottie barnes": "Scottie Barnes",
    "evan mobley": "Evan Mobley",
    "cade cunningham": "Cade Cunningham",
    "ant edwards": "Anthony Edwards",
    "anthony edwards": "Anthony Edwards",
    "ant man": "Anthony Edwards",
    "luka garza": "Luka Garza",
    "franz wagner": "Franz Wagner",

    # Nombres con ' (apóstrofe)
    "d'angelo russell": "D'Angelo Russell",
    "dangelo russell": "D'Angelo Russell",

    # Más Jr/III
    "walker kessler": "Walker Kessler",
    "jabari walker": "Jabari Walker",
    "mark williams": "Mark Williams",
    "jalen green": "Jalen Green",
}


def normalize_name(name: str) -> str:
    """
    Normaliza un nombre para comparación:
    - Lowercase
    - Remove acentos
    - Remove puntuación extra
    - Normalizar espacios
    """
    if not name:
        return ""

    # Lowercase
    name = name.lower().strip()

    # Normalizar unicode (remover acentos)
    name = unicodedata.normalize('NFKD', name)
    name = ''.join(c for c in name if not unicodedata.combining(c))

    # Remover puntuación excepto apóstrofes y guiones
    name = re.sub(r"[^\w\s'-]", "", name)

    # Normalizar espacios múltiples
    name = re.sub(r"\s+", " ", name)

    return name.strip()


class PlayerNameMatcher:
    """
    Sistema de resolución de nombres de jugadores.

    Usa una combinación de:
    1. Mapeos manuales para casos conocidos
    2. Fuzzy matching para variaciones menores
    3. Cache para evitar recálculos
    """

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DB_PATH
        self.canonical_names: Dict[str, int] = {}  # name -> player_id
        self.normalized_to_canonical: Dict[str, str] = {}  # normalized -> canonical
        self.cache: Dict[str, Optional[Tuple[str, int]]] = {}  # scraper_name -> (canonical, id)
        self.unmatched: set = set()  # Para logging de nombres no encontrados

        self._load_players()

    def _load_players(self):
        """Carga jugadores de la base de datos."""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql("""
                SELECT DISTINCT player_id, player_name
                FROM player_game_logs
                WHERE player_name IS NOT NULL
            """, conn)
            conn.close()

            for _, row in df.iterrows():
                name = row["player_name"]
                player_id = row["player_id"]
                self.canonical_names[name] = player_id
                self.normalized_to_canonical[normalize_name(name)] = name

            print(f"PlayerNameMatcher: Cargados {len(self.canonical_names)} jugadores")

        except Exception as e:
            print(f"Error cargando jugadores: {e}")

    def resolve(self, scraper_name: str, min_score: int = 85) -> Optional[Tuple[str, int]]:
        """
        Resuelve un nombre de scraper al nombre canónico y player_id.

        Args:
            scraper_name: Nombre como viene del scraper (DraftKings, FanDuel, etc.)
            min_score: Score mínimo de fuzzy matching (0-100)

        Returns:
            Tuple[canonical_name, player_id] o None si no se encuentra
        """
        if not scraper_name:
            return None

        # Check cache
        if scraper_name in self.cache:
            return self.cache[scraper_name]

        # Normalizar
        normalized = normalize_name(scraper_name)

        # 1. Buscar en mapeos manuales
        if normalized in MANUAL_MAPPINGS:
            canonical = MANUAL_MAPPINGS[normalized]
            if canonical in self.canonical_names:
                result = (canonical, self.canonical_names[canonical])
                self.cache[scraper_name] = result
                return result

        # 2. Buscar match exacto normalizado
        if normalized in self.normalized_to_canonical:
            canonical = self.normalized_to_canonical[normalized]
            result = (canonical, self.canonical_names[canonical])
            self.cache[scraper_name] = result
            return result

        # 3. Fuzzy matching
        if FUZZY_AVAILABLE and self.normalized_to_canonical:
            match = process.extractOne(
                normalized,
                list(self.normalized_to_canonical.keys()),
                scorer=fuzz.ratio
            )

            if match and match[1] >= min_score:
                canonical = self.normalized_to_canonical[match[0]]
                result = (canonical, self.canonical_names[canonical])
                self.cache[scraper_name] = result
                return result

        # No match found
        self.cache[scraper_name] = None
        self.unmatched.add(scraper_name)
        return None

    def get_canonical_name(self, scraper_name: str) -> Optional[str]:
        """Devuelve solo el nombre canónico."""
        result = self.resolve(scraper_name)
        return result[0] if result else None

    def get_player_id(self, scraper_name: str) -> Optional[int]:
        """Devuelve solo el player_id."""
        result = self.resolve(scraper_name)
        return result[1] if result else None

    def get_unmatched_names(self) -> set:
        """Devuelve nombres que no se pudieron resolver (para logging/debugging)."""
        return self.unmatched

    def find_best_match(self, player_name: str, candidates: list,
                        threshold: int = 85) -> Optional[str]:
        """
        Encuentra el mejor match para un nombre en una lista de candidatos.

        Útil para matching contra listas dinámicas (ej: props de PrizePicks).

        Args:
            player_name: Nombre a buscar
            candidates: Lista de nombres candidatos
            threshold: Score mínimo de similarity (0-100)

        Returns:
            El mejor match si supera el threshold, None otherwise
        """
        if not player_name or not candidates:
            return None

        normalized_input = normalize_name(player_name)

        # 1. Primero buscar exact match (normalizado)
        for candidate in candidates:
            if normalize_name(candidate) == normalized_input:
                return candidate

        # 2. Buscar en mapeos manuales
        if normalized_input in MANUAL_MAPPINGS:
            canonical = MANUAL_MAPPINGS[normalized_input]
            for candidate in candidates:
                if normalize_name(candidate) == normalize_name(canonical):
                    return candidate

        # 3. Fuzzy matching contra la lista de candidatos
        if FUZZY_AVAILABLE:
            # Normalizar candidatos para comparación
            normalized_candidates = {normalize_name(c): c for c in candidates}

            match = process.extractOne(
                normalized_input,
                list(normalized_candidates.keys()),
                scorer=fuzz.ratio
            )

            if match and match[1] >= threshold:
                return normalized_candidates[match[0]]

        return None

    def add_manual_mapping(self, scraper_name: str, canonical_name: str):
        """Añade un mapeo manual en runtime."""
        normalized = normalize_name(scraper_name)
        MANUAL_MAPPINGS[normalized] = canonical_name
        # Limpiar cache para este nombre
        if scraper_name in self.cache:
            del self.cache[scraper_name]


# Singleton global para reusar en todo el sistema
_matcher_instance: Optional[PlayerNameMatcher] = None


def get_matcher() -> PlayerNameMatcher:
    """Obtiene la instancia singleton del matcher."""
    global _matcher_instance
    if _matcher_instance is None:
        _matcher_instance = PlayerNameMatcher()
    return _matcher_instance


def resolve_player_name(scraper_name: str) -> Optional[str]:
    """Función de conveniencia para resolver un nombre."""
    return get_matcher().get_canonical_name(scraper_name)


def resolve_player_id(scraper_name: str) -> Optional[int]:
    """Función de conveniencia para obtener player_id."""
    return get_matcher().get_player_id(scraper_name)


if __name__ == "__main__":
    # Test del matcher
    matcher = PlayerNameMatcher()

    test_names = [
        "LeBron James",
        "Nikola Jokic",  # Sin acento
        "Nikola Jokić",  # Con acento
        "Jaren Jackson Jr",  # Sin punto
        "Jaren Jackson Jr.",  # Con punto
        "Nic Claxton",  # Apodo
        "Nicolas Claxton",  # Nombre completo
        "Dennis Schroder",  # Sin umlaut
        "P.J. Washington",
        "PJ Washington",
        "Shai Gilgeous-Alexander",
        "Shai Gilgeous Alexander",  # Sin guión
        "Cam Thomas",  # Nombre corto
        "Ant Edwards",  # Apodo
        "asdfasdf",  # Nombre inventado (no debería matchear)
    ]

    print("\n=== Test de Name Matching (resolve) ===\n")

    for name in test_names:
        result = matcher.resolve(name)
        if result:
            print(f"✅ '{name}' -> '{result[0]}' (ID: {result[1]})")
        else:
            print(f"❌ '{name}' -> NO MATCH")

    print(f"\n⚠️ Nombres sin match: {matcher.get_unmatched_names()}")

    # Test find_best_match (para PrizePicks/odds matching)
    print("\n=== Test de find_best_match (odds matching) ===\n")

    # Simular lista de props de PrizePicks
    prizepicks_names = [
        "Nicolas Claxton",
        "P.J. Washington",
        "Shai Gilgeous-Alexander",
        "Anthony Edwards",
        "Cameron Thomas",
    ]

    test_queries = [
        "Nic Claxton",       # -> Nicolas Claxton
        "PJ Washington",     # -> P.J. Washington
        "Shai Alexander",    # -> Shai Gilgeous-Alexander
        "Ant Edwards",       # -> Anthony Edwards
        "Cam Thomas",        # -> Cameron Thomas
        "Fake Player",       # -> None
    ]

    for query in test_queries:
        match = matcher.find_best_match(query, prizepicks_names, threshold=85)
        if match:
            print(f"✅ '{query}' -> '{match}'")
        else:
            print(f"❌ '{query}' -> NO MATCH")
