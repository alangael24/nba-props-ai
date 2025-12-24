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
    TeamGameLog,
    BoxScoreTraditionalV2
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

    # Tabla para contexto de compañeros (Teammates Available)
    # Trackea si los jugadores clave del equipo estuvieron disponibles
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS teammate_context (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL,
            game_id TEXT NOT NULL,
            game_date DATE NOT NULL,
            team_id INTEGER NOT NULL,
            -- Métricas de disponibilidad de compañeros
            star1_available INTEGER DEFAULT 1,  -- 1 = jugó, 0 = no jugó
            star2_available INTEGER DEFAULT 1,
            star3_available INTEGER DEFAULT 1,
            total_stars_available INTEGER DEFAULT 3,
            -- Contexto adicional
            primary_pg_available INTEGER DEFAULT 1,  -- Point guard principal
            primary_center_available INTEGER DEFAULT 1,  -- Center principal
            team_minutes_load REAL,  -- Total minutos del equipo (detecta blowouts)
            -- Para cálculo de usage boost
            usage_boost_expected REAL DEFAULT 0,  -- Boost esperado si faltan estrellas
            UNIQUE(player_id, game_id)
        )
    """)

    # Tabla para identificar jugadores clave por equipo
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS team_key_players (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id INTEGER NOT NULL,
            team_abbrev TEXT,
            season TEXT NOT NULL,
            -- Top 3 jugadores por scoring
            star1_player_id INTEGER,
            star1_name TEXT,
            star1_ppg REAL,
            star2_player_id INTEGER,
            star2_name TEXT,
            star2_ppg REAL,
            star3_player_id INTEGER,
            star3_name TEXT,
            star3_ppg REAL,
            -- Rol específico
            primary_pg_id INTEGER,
            primary_pg_name TEXT,
            primary_center_id INTEGER,
            primary_center_name TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(team_id, season)
        )
    """)

    # Índices para consultas rápidas
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_game_logs_player ON player_game_logs(player_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_game_logs_date ON player_game_logs(game_date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_game_logs_opponent ON player_game_logs(opponent_team_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_game_logs_game_id ON player_game_logs(game_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_teammate_context_player ON teammate_context(player_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_teammate_context_game ON teammate_context(game_id)")

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


def get_team_id_mapping(conn: sqlite3.Connection) -> dict:
    """Obtiene mapeo de abreviatura de equipo a team_id."""
    teams_df = pd.read_sql("SELECT team_id, abbreviation FROM teams", conn)
    return dict(zip(teams_df["abbreviation"], teams_df["team_id"]))


def save_game_logs(df: pd.DataFrame):
    """
    Guarda los game logs en la base de datos usando INSERT OR IGNORE.
    También completa opponent_team_id.
    """
    if df.empty:
        return 0

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Obtener mapeo de equipos
    team_mapping = get_team_id_mapping(conn)

    # Completar opponent_team_id
    df["opponent_team_id"] = df["opponent_abbrev"].map(team_mapping)

    # Insertar fila por fila con INSERT OR IGNORE
    columns = [
        "player_id", "player_name", "season", "game_id", "game_date",
        "matchup", "is_home", "wl", "min", "pts", "reb", "ast",
        "stl", "blk", "tov", "fgm", "fga", "fg_pct", "fg3m", "fg3a",
        "fg3_pct", "ftm", "fta", "ft_pct", "oreb", "dreb", "pf",
        "plus_minus", "opponent_team_id", "opponent_abbrev"
    ]

    # Filtrar solo columnas que existen
    available_cols = [c for c in columns if c in df.columns]

    inserted = 0
    for _, row in df.iterrows():
        values = [row.get(c) for c in available_cols]
        placeholders = ", ".join(["?"] * len(available_cols))
        col_names = ", ".join(available_cols)

        try:
            cursor.execute(
                f"INSERT OR IGNORE INTO player_game_logs ({col_names}) VALUES ({placeholders})",
                values
            )
            if cursor.rowcount > 0:
                inserted += 1
        except sqlite3.Error as e:
            # Log error pero continuar
            continue

    conn.commit()
    conn.close()
    return inserted


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


# =============================================================================
# TEAMMATE CONTEXT - Ingesta de contexto de compañeros
# =============================================================================

def identify_team_key_players(season: str):
    """
    Identifica los jugadores clave de cada equipo basándose en los game logs.

    Para cada equipo encuentra:
    - Top 3 anotadores (estrellas)
    - Point guard principal (más asistencias)
    - Center principal (más rebotes)
    """
    print(f"\n[TEAMMATES] Identificando jugadores clave para {season}...")

    conn = sqlite3.connect(DB_PATH)

    # Obtener promedios por jugador/equipo
    query = """
        SELECT
            p.player_id,
            p.player_name,
            p.team_id,
            t.abbreviation as team_abbrev,
            AVG(g.pts) as ppg,
            AVG(g.reb) as rpg,
            AVG(g.ast) as apg,
            AVG(g.min) as mpg,
            COUNT(*) as games
        FROM player_game_logs g
        JOIN players p ON g.player_id = p.player_id
        JOIN teams t ON p.team_id = t.team_id
        WHERE g.season = ?
            AND g.min >= 15
        GROUP BY g.player_id
        HAVING games >= 10
    """

    df = pd.read_sql(query, conn, params=[season])

    if df.empty:
        print("  No hay suficientes datos para identificar jugadores clave")
        conn.close()
        return

    teams_processed = 0

    # Para cada equipo, identificar jugadores clave
    for team_id in df["team_id"].unique():
        team_df = df[df["team_id"] == team_id].copy()
        team_abbrev = team_df["team_abbrev"].iloc[0]

        if len(team_df) < 3:
            continue

        # Top 3 por puntos (estrellas)
        top_scorers = team_df.nlargest(3, "ppg")

        # Point guard (más asistencias entre los que promedian >3 apg)
        pgs = team_df[team_df["apg"] >= 3].nlargest(1, "apg")
        primary_pg = pgs.iloc[0] if len(pgs) > 0 else None

        # Center (más rebotes entre los que promedian >5 rpg)
        centers = team_df[team_df["rpg"] >= 5].nlargest(1, "rpg")
        primary_center = centers.iloc[0] if len(centers) > 0 else None

        # Insertar en team_key_players
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO team_key_players
            (team_id, team_abbrev, season,
             star1_player_id, star1_name, star1_ppg,
             star2_player_id, star2_name, star2_ppg,
             star3_player_id, star3_name, star3_ppg,
             primary_pg_id, primary_pg_name,
             primary_center_id, primary_center_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            int(team_id), team_abbrev, season,
            int(top_scorers.iloc[0]["player_id"]), top_scorers.iloc[0]["player_name"], round(top_scorers.iloc[0]["ppg"], 1),
            int(top_scorers.iloc[1]["player_id"]), top_scorers.iloc[1]["player_name"], round(top_scorers.iloc[1]["ppg"], 1),
            int(top_scorers.iloc[2]["player_id"]) if len(top_scorers) > 2 else None,
            top_scorers.iloc[2]["player_name"] if len(top_scorers) > 2 else None,
            round(top_scorers.iloc[2]["ppg"], 1) if len(top_scorers) > 2 else None,
            int(primary_pg["player_id"]) if primary_pg is not None else None,
            primary_pg["player_name"] if primary_pg is not None else None,
            int(primary_center["player_id"]) if primary_center is not None else None,
            primary_center["player_name"] if primary_center is not None else None
        ))

        teams_processed += 1

    conn.commit()
    conn.close()
    print(f"  ✅ Identificados jugadores clave para {teams_processed} equipos")


def build_teammate_context(season: str):
    """
    Construye el contexto de compañeros para cada partido.

    Para cada jugador en cada partido, determina:
    - ¿Estaban jugando las estrellas del equipo?
    - ¿Estaba el PG/C principal?
    - ¿Cuánto usage boost se espera si faltan estrellas?
    """
    print(f"\n[TEAMMATES] Construyendo contexto de compañeros para {season}...")

    conn = sqlite3.connect(DB_PATH)

    # Obtener jugadores clave por equipo
    key_players_df = pd.read_sql("""
        SELECT * FROM team_key_players WHERE season = ?
    """, conn, params=[season])

    if key_players_df.empty:
        print("  ⚠️ No hay jugadores clave identificados. Ejecuta --identify-stars primero.")
        conn.close()
        return

    # Crear diccionario de key players por team_id
    key_players = {}
    for _, row in key_players_df.iterrows():
        key_players[row["team_id"]] = {
            "stars": [row["star1_player_id"], row["star2_player_id"], row["star3_player_id"]],
            "star_ppg": [row["star1_ppg"], row["star2_ppg"], row["star3_ppg"]],
            "pg": row["primary_pg_id"],
            "center": row["primary_center_id"]
        }

    # Obtener todos los game_ids únicos
    games_df = pd.read_sql("""
        SELECT DISTINCT game_id, game_date
        FROM player_game_logs
        WHERE season = ?
        ORDER BY game_date
    """, conn, params=[season])

    print(f"  Procesando {len(games_df)} partidos...")

    inserted = 0

    for _, game in tqdm(games_df.iterrows(), total=len(games_df), desc="  Contexto"):
        game_id = game["game_id"]
        game_date = game["game_date"]

        # Obtener todos los jugadores que jugaron en este partido
        players_in_game = pd.read_sql("""
            SELECT player_id, player_name, min
            FROM player_game_logs
            WHERE game_id = ? AND min > 0
        """, conn, params=[game_id])

        players_set = set(players_in_game["player_id"].tolist())

        # Para cada jugador que jugó, calcular contexto
        for _, player_row in players_in_game.iterrows():
            player_id = player_row["player_id"]

            # Obtener equipo del jugador
            team_query = pd.read_sql("""
                SELECT team_id FROM players WHERE player_id = ?
            """, conn, params=[int(player_id)])

            if team_query.empty:
                continue

            team_id = team_query.iloc[0]["team_id"]

            if team_id not in key_players:
                continue

            kp = key_players[team_id]

            # Calcular disponibilidad de estrellas (excluyendo al jugador mismo)
            stars = [s for s in kp["stars"] if s is not None and s != player_id]
            star_ppgs = [ppg for s, ppg in zip(kp["stars"], kp["star_ppg"])
                        if s is not None and s != player_id and ppg is not None]

            star1_available = 1 if len(stars) > 0 and stars[0] in players_set else 0
            star2_available = 1 if len(stars) > 1 and stars[1] in players_set else 0
            star3_available = 1 if len(stars) > 2 and stars[2] in players_set else 0
            total_stars = star1_available + star2_available + star3_available

            # PG y Center
            pg_available = 1 if kp["pg"] is None or kp["pg"] == player_id or kp["pg"] in players_set else 0
            center_available = 1 if kp["center"] is None or kp["center"] == player_id or kp["center"] in players_set else 0

            # Calcular usage boost esperado
            # Si falta una estrella de 25 PPG, se reparte entre los demás
            usage_boost = 0.0
            missing_ppg = 0
            if len(stars) > 0 and star1_available == 0 and len(star_ppgs) > 0:
                missing_ppg += star_ppgs[0]
            if len(stars) > 1 and star2_available == 0 and len(star_ppgs) > 1:
                missing_ppg += star_ppgs[1]

            # Usage boost = % del PPG perdido que podría absorber este jugador
            # Simplificación: 20% del PPG perdido se reparte
            if missing_ppg > 0:
                usage_boost = missing_ppg * 0.20

            # Insertar contexto
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR IGNORE INTO teammate_context
                    (player_id, game_id, game_date, team_id,
                     star1_available, star2_available, star3_available, total_stars_available,
                     primary_pg_available, primary_center_available, usage_boost_expected)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    int(player_id), game_id, game_date, int(team_id),
                    star1_available, star2_available, star3_available, total_stars,
                    pg_available, center_available, round(usage_boost, 2)
                ))
                if cursor.rowcount > 0:
                    inserted += 1
            except sqlite3.Error:
                continue

    conn.commit()
    conn.close()
    print(f"  ✅ Guardados {inserted} registros de contexto de compañeros")


def run_teammate_analysis():
    """Ejecuta el análisis completo de teammates."""
    print("\n" + "=" * 60)
    print("ANÁLISIS DE CONTEXTO DE COMPAÑEROS (TEAMMATES)")
    print("=" * 60)

    # Asegurar que las tablas existen
    create_database()

    for season in SEASONS:
        # Paso 1: Identificar jugadores clave
        identify_team_key_players(season)

        # Paso 2: Construir contexto de compañeros
        build_teammate_context(season)

    # Mostrar resumen
    conn = sqlite3.connect(DB_PATH)

    print("\n--- RESUMEN DE KEY PLAYERS ---")
    key_players = pd.read_sql("""
        SELECT team_abbrev, season, star1_name, star1_ppg, star2_name, star2_ppg
        FROM team_key_players
        ORDER BY season DESC, star1_ppg DESC
        LIMIT 10
    """, conn)
    print(key_players.to_string(index=False))

    print("\n--- MUESTRA DE CONTEXTO ---")
    context_sample = pd.read_sql("""
        SELECT
            tc.game_date,
            p.player_name,
            tc.total_stars_available,
            tc.usage_boost_expected
        FROM teammate_context tc
        JOIN players p ON tc.player_id = p.player_id
        WHERE tc.usage_boost_expected > 0
        ORDER BY tc.usage_boost_expected DESC
        LIMIT 10
    """, conn)
    print(context_sample.to_string(index=False))

    conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingesta de datos NBA")
    parser.add_argument("--stats", action="store_true", help="Mostrar estadísticas de la DB")
    parser.add_argument("--quick", action="store_true", help="Solo descargar temporada actual")
    parser.add_argument("--teammates", action="store_true",
                        help="Construir contexto de compañeros (teammates available)")

    args = parser.parse_args()

    if args.stats:
        quick_stats()
    elif args.teammates:
        if args.quick:
            SEASONS = ["2024-25"]
        run_teammate_analysis()
    else:
        if args.quick:
            SEASONS = ["2024-25"]
        run_full_ingestion()
