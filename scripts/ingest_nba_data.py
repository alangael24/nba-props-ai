"""
Fase 1.2: Ingesta de datos históricos de la NBA
Descarga Game Logs de todos los jugadores de las últimas N temporadas.
"""

import sqlite3
import time
import random
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# nba_api imports
from nba_api.stats.endpoints import (
    PlayerGameLog,
    CommonAllPlayers,
    LeagueDashTeamStats,
    TeamGameLog
)
from nba_api.stats.static import teams

# Configuración
DB_PATH = Path(__file__).parent.parent / "data" / "nba_props.db"
SEASONS = ["2022-23", "2023-24", "2024-25"]  # Últimas 3 temporadas
REQUEST_DELAY = 2.0  # Delay base entre requests
REQUEST_JITTER = 1.0  # Jitter aleatorio adicional (0-1 seg)
MAX_RETRIES = 3  # Reintentos por jugador
RETRY_DELAY = 45  # Segundos base entre reintentos


def create_database():
    """Crea las tablas necesarias en SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Tabla de jugadores
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS players (
            player_id INTEGER PRIMARY KEY,
            player_name TEXT NOT NULL,
            team_id INTEGER,
            team_abbreviation TEXT,
            is_active INTEGER,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Tabla de Game Logs (el oro del sistema)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS player_game_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL,
            player_name TEXT,
            season TEXT NOT NULL,
            game_id TEXT NOT NULL,
            game_date DATE NOT NULL,
            matchup TEXT,
            is_home INTEGER,
            wl TEXT,
            min REAL,
            pts INTEGER,
            reb INTEGER,
            ast INTEGER,
            stl INTEGER,
            blk INTEGER,
            tov INTEGER,
            fgm INTEGER,
            fga INTEGER,
            fg_pct REAL,
            fg3m INTEGER,
            fg3a INTEGER,
            fg3_pct REAL,
            ftm INTEGER,
            fta INTEGER,
            ft_pct REAL,
            oreb INTEGER,
            dreb INTEGER,
            pf INTEGER,
            plus_minus INTEGER,
            opponent_team_id INTEGER,
            opponent_abbrev TEXT,
            UNIQUE(player_id, game_id)
        )
    """)

    # Tabla de equipos
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS teams (
            team_id INTEGER PRIMARY KEY,
            team_name TEXT NOT NULL,
            abbreviation TEXT NOT NULL,
            conference TEXT,
            division TEXT
        )
    """)

    # Tabla para Defense vs Position (DvP) - Fase 1.3
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS defense_vs_position (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id INTEGER NOT NULL,
            team_abbrev TEXT,
            season TEXT NOT NULL,
            position TEXT NOT NULL,
            pts_allowed_avg REAL,
            reb_allowed_avg REAL,
            ast_allowed_avg REAL,
            games_sample INTEGER,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(team_id, season, position)
        )
    """)

    # Índices para consultas rápidas
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_game_logs_player ON player_game_logs(player_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_game_logs_date ON player_game_logs(game_date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_game_logs_opponent ON player_game_logs(opponent_team_id)")

    conn.commit()
    conn.close()
    print(f"Base de datos creada en: {DB_PATH}")


def get_all_nba_teams():
    """Obtiene todos los equipos de la NBA y los guarda."""
    conn = sqlite3.connect(DB_PATH)
    nba_teams = teams.get_teams()

    df_teams = pd.DataFrame(nba_teams)
    df_teams = df_teams.rename(columns={
        "id": "team_id",
        "full_name": "team_name",
        "abbreviation": "abbreviation"
    })

    # Solo columnas necesarias
    df_teams = df_teams[["team_id", "team_name", "abbreviation"]]
    df_teams["conference"] = None
    df_teams["division"] = None

    df_teams.to_sql("teams", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Guardados {len(df_teams)} equipos")
    return df_teams


def get_active_players(season: str):
    """Obtiene lista de jugadores activos para una temporada."""
    try:
        # Usar is_only_current_season=0 para obtener TODOS los jugadores
        all_players = CommonAllPlayers(
            is_only_current_season=0,
            league_id="00",
            season=season
        )
        time.sleep(REQUEST_DELAY)

        df = all_players.get_data_frames()[0]

        # Filtrar jugadores que tienen equipo asignado (están activos)
        df = df[df["TEAM_ID"].notna() & (df["TEAM_ID"] != 0)]

        return df[["PERSON_ID", "DISPLAY_FIRST_LAST", "TEAM_ID", "TEAM_ABBREVIATION"]].rename(columns={
            "PERSON_ID": "player_id",
            "DISPLAY_FIRST_LAST": "player_name",
            "TEAM_ID": "team_id",
            "TEAM_ABBREVIATION": "team_abbreviation"
        })
    except Exception as e:
        print(f"Error obteniendo jugadores: {e}")
        return pd.DataFrame()


def extract_opponent_info(matchup: str, is_home: bool):
    """Extrae el equipo rival del string de matchup."""
    if not matchup:
        return None, None

    # Formato: "LAL vs. GSW" o "LAL @ GSW"
    if " vs. " in matchup:
        parts = matchup.split(" vs. ")
        opponent = parts[1].strip()
    elif " @ " in matchup:
        parts = matchup.split(" @ ")
        opponent = parts[1].strip()
    else:
        return None, None

    return opponent, None  # team_id se puede resolver después


def download_player_game_logs(player_id: int, player_name: str, season: str):
    """Descarga los game logs de un jugador para una temporada con reintentos."""
    for attempt in range(MAX_RETRIES):
        try:
            game_log = PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star="Regular Season"
            )

            df = game_log.get_data_frames()[0]

            # Delay con jitter para evitar rate limiting
            time.sleep(REQUEST_DELAY + random.uniform(0, REQUEST_JITTER))

            if df.empty:
                return pd.DataFrame()

            # Procesar datos
            df["season"] = season
            df["player_name"] = player_name  # Asignar nombre desde parámetro
            df["is_home"] = df["MATCHUP"].apply(lambda x: 1 if " vs. " in str(x) else 0)
            df["opponent_abbrev"] = df["MATCHUP"].apply(
                lambda x: x.split(" vs. ")[-1].strip() if " vs. " in str(x)
                else x.split(" @ ")[-1].strip() if " @ " in str(x) else None
            )

            # Renombrar columnas
            df = df.rename(columns={
                "Player_ID": "player_id",
                "SEASON_ID": "season_id",
                "Game_ID": "game_id",
                "GAME_DATE": "game_date",
                "MATCHUP": "matchup",
                "WL": "wl",
                "MIN": "min",
                "PTS": "pts",
                "REB": "reb",
                "AST": "ast",
                "STL": "stl",
                "BLK": "blk",
                "TOV": "tov",
                "FGM": "fgm",
                "FGA": "fga",
                "FG_PCT": "fg_pct",
                "FG3M": "fg3m",
                "FG3A": "fg3a",
                "FG3_PCT": "fg3_pct",
                "FTM": "ftm",
                "FTA": "fta",
                "FT_PCT": "ft_pct",
                "OREB": "oreb",
                "DREB": "dreb",
                "PF": "pf",
                "PLUS_MINUS": "plus_minus"
            })

            # Seleccionar columnas finales
            columns = [
                "player_id", "player_name", "season", "game_id", "game_date",
                "matchup", "is_home", "wl", "min", "pts", "reb", "ast",
                "stl", "blk", "tov", "fgm", "fga", "fg_pct", "fg3m", "fg3a",
                "fg3_pct", "ftm", "fta", "ft_pct", "oreb", "dreb", "pf",
                "plus_minus", "opponent_abbrev"
            ]

            return df[[c for c in columns if c in df.columns]]

        except Exception as e:
            wait_time = RETRY_DELAY * (2 ** attempt)  # Backoff exponencial: 30, 60, 120 seg
            if attempt < MAX_RETRIES - 1:
                print(f"  ⚠️ Timeout {player_name} (intento {attempt + 1}/{MAX_RETRIES}), esperando {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  ❌ Error final con {player_name}: {e}")
                return pd.DataFrame()

    return pd.DataFrame()


def save_game_logs(df: pd.DataFrame):
    """Guarda los game logs en la base de datos."""
    if df.empty:
        return 0

    conn = sqlite3.connect(DB_PATH)

    # Insert or replace para evitar duplicados
    df.to_sql("player_game_logs", conn, if_exists="append", index=False)

    conn.close()
    return len(df)


def get_processed_players(season: str) -> set:
    """Obtiene IDs de jugadores que ya tienen datos para una temporada."""
    if not DB_PATH.exists():
        return set()

    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql(f"""
            SELECT DISTINCT player_id
            FROM player_game_logs
            WHERE season = '{season}'
        """, conn)
        return set(df["player_id"].tolist())
    except Exception:
        return set()
    finally:
        conn.close()


def run_full_ingestion():
    """Ejecuta la ingesta completa de datos."""
    print("=" * 60)
    print("SISTEMA DE PREDICCIÓN NBA PROPS - INGESTA DE DATOS")
    print("=" * 60)

    # Crear DB
    create_database()

    # Guardar equipos
    print("\n[1/3] Obteniendo equipos NBA...")
    get_all_nba_teams()

    # Estadísticas globales
    total_games = 0
    total_players = 0

    for season in SEASONS:
        print(f"\n[2/3] Procesando temporada {season}...")

        # Obtener jugadores activos
        players_df = get_active_players(season)
        if players_df.empty:
            print(f"  No se encontraron jugadores para {season}")
            continue

        print(f"  Encontrados {len(players_df)} jugadores activos")

        # Verificar jugadores ya procesados (para resumir)
        processed = get_processed_players(season)
        pending_players = players_df[~players_df["player_id"].isin(processed)]

        if len(processed) > 0:
            print(f"  ✅ Ya procesados: {len(processed)} jugadores")
            print(f"  ⏳ Pendientes: {len(pending_players)} jugadores")

        # Guardar jugadores
        conn = sqlite3.connect(DB_PATH)
        players_df["is_active"] = 1
        players_df.to_sql("players", conn, if_exists="replace", index=False)
        conn.close()

        # Descargar game logs solo de jugadores pendientes
        season_games = 0

        for _, player in tqdm(pending_players.iterrows(), total=len(pending_players), desc=f"  Descargando {season}"):
            game_logs = download_player_game_logs(
                player["player_id"],
                player["player_name"],
                season
            )

            if not game_logs.empty:
                try:
                    saved = save_game_logs(game_logs)
                    season_games += saved
                except Exception as e:
                    # Probablemente duplicados, ignorar
                    pass

        print(f"  Guardados {season_games} registros de partidos")
        total_games += season_games
        total_players += len(players_df)

    # Resumen final
    print("\n" + "=" * 60)
    print("INGESTA COMPLETADA")
    print("=" * 60)
    print(f"Temporadas procesadas: {len(SEASONS)}")
    print(f"Total jugadores: {total_players}")
    print(f"Total registros de partidos: {total_games}")
    print(f"Base de datos: {DB_PATH}")

    # Verificar datos
    conn = sqlite3.connect(DB_PATH)
    sample = pd.read_sql("SELECT * FROM player_game_logs LIMIT 5", conn)
    conn.close()

    print("\nMuestra de datos guardados:")
    print(sample[["player_name", "game_date", "matchup", "pts", "reb", "ast"]].to_string())


def quick_stats():
    """Muestra estadísticas rápidas de la base de datos."""
    conn = sqlite3.connect(DB_PATH)

    print("\n--- ESTADÍSTICAS DE LA BASE DE DATOS ---")

    # Total de registros
    total = pd.read_sql("SELECT COUNT(*) as total FROM player_game_logs", conn)
    print(f"Total de game logs: {total['total'].values[0]}")

    # Por temporada
    by_season = pd.read_sql("""
        SELECT season, COUNT(*) as games
        FROM player_game_logs
        GROUP BY season
    """, conn)
    print(f"\nPor temporada:\n{by_season.to_string(index=False)}")

    # Top scorers
    top_scorers = pd.read_sql("""
        SELECT player_name, ROUND(AVG(pts), 1) as ppg, COUNT(*) as games
        FROM player_game_logs
        WHERE min > 20
        GROUP BY player_id
        HAVING games > 50
        ORDER BY ppg DESC
        LIMIT 10
    """, conn)
    print(f"\nTop 10 anotadores (min 50 partidos, >20 min):\n{top_scorers.to_string(index=False)}")

    conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingesta de datos NBA")
    parser.add_argument("--stats", action="store_true", help="Mostrar estadísticas de la DB")
    parser.add_argument("--quick", action="store_true", help="Solo descargar temporada actual")

    args = parser.parse_args()

    if args.stats:
        quick_stats()
    else:
        if args.quick:
            SEASONS = ["2024-25"]
        run_full_ingestion()
