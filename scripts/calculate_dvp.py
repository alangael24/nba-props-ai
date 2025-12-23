"""
Fase 1.3: Cálculo de Defense vs Position (DvP)
Cuántos puntos/rebotes/asistencias permite cada equipo a cada posición.
"""

import sqlite3
from pathlib import Path

import pandas as pd
import numpy as np

DB_PATH = Path(__file__).parent.parent / "data" / "nba_props.db"

# Mapeo de jugadores a posiciones (simplificado)
# En producción, usar la API para obtener posiciones reales
POSITION_MAPPING = {
    # Guards
    "PG": ["Stephen Curry", "Luka Doncic", "Trae Young", "Ja Morant", "Tyrese Haliburton",
           "Damian Lillard", "Shai Gilgeous-Alexander", "De'Aaron Fox", "Jalen Brunson"],
    "SG": ["Devin Booker", "Donovan Mitchell", "Anthony Edwards", "Jaylen Brown",
           "Zach LaVine", "Bradley Beal", "CJ McCollum", "Desmond Bane"],
    # Forwards
    "SF": ["LeBron James", "Kevin Durant", "Jayson Tatum", "Jimmy Butler", "Paul George",
           "Kawhi Leonard", "Brandon Ingram", "DeMar DeRozan", "Khris Middleton"],
    "PF": ["Giannis Antetokounmpo", "Lauri Markkanen", "Jaren Jackson Jr.", "Pascal Siakam",
           "Julius Randle", "Scottie Barnes", "Draymond Green", "Tobias Harris"],
    # Centers
    "C": ["Nikola Jokic", "Joel Embiid", "Anthony Davis", "Bam Adebayo", "Rudy Gobert",
          "Karl-Anthony Towns", "Domantas Sabonis", "Chet Holmgren", "Victor Wembanyama"]
}


def get_player_position(player_name: str) -> str:
    """Obtiene la posición de un jugador."""
    for position, players in POSITION_MAPPING.items():
        if player_name in players:
            return position

    # Por defecto, asignar basándose en estadísticas
    # (más rebotes = big, más asistencias = guard)
    return "UNKNOWN"


def calculate_dvp_from_game_logs():
    """
    Calcula DvP basándose en los game logs existentes.
    Agrupa por equipo rival y calcula promedios permitidos.
    """
    conn = sqlite3.connect(DB_PATH)

    print("Calculando Defense vs Position...")

    # Obtener todos los game logs con oponente
    query = """
        SELECT
            player_name,
            opponent_abbrev,
            season,
            pts,
            reb,
            ast,
            min
        FROM player_game_logs
        WHERE min > 15  -- Solo jugadores con minutos significativos
        AND opponent_abbrev IS NOT NULL
    """

    df = pd.read_sql(query, conn)

    if df.empty:
        print("No hay datos suficientes para calcular DvP")
        return

    # Asignar posiciones
    df["position"] = df["player_name"].apply(get_player_position)

    # Filtrar jugadores con posición conocida
    df_known = df[df["position"] != "UNKNOWN"]

    print(f"Procesando {len(df_known)} registros con posición conocida")

    # Calcular DvP: promedios que permite cada equipo a cada posición
    dvp_stats = df_known.groupby(["opponent_abbrev", "season", "position"]).agg({
        "pts": "mean",
        "reb": "mean",
        "ast": "mean",
        "player_name": "count"  # número de partidos como muestra
    }).reset_index()

    dvp_stats = dvp_stats.rename(columns={
        "opponent_abbrev": "team_abbrev",
        "pts": "pts_allowed_avg",
        "reb": "reb_allowed_avg",
        "ast": "ast_allowed_avg",
        "player_name": "games_sample"
    })

    # Redondear
    dvp_stats["pts_allowed_avg"] = dvp_stats["pts_allowed_avg"].round(1)
    dvp_stats["reb_allowed_avg"] = dvp_stats["reb_allowed_avg"].round(1)
    dvp_stats["ast_allowed_avg"] = dvp_stats["ast_allowed_avg"].round(1)

    # Obtener team_id
    teams_df = pd.read_sql("SELECT team_id, abbreviation FROM teams", conn)
    dvp_stats = dvp_stats.merge(
        teams_df,
        left_on="team_abbrev",
        right_on="abbreviation",
        how="left"
    )

    # Guardar en DB
    dvp_stats_final = dvp_stats[[
        "team_id", "team_abbrev", "season", "position",
        "pts_allowed_avg", "reb_allowed_avg", "ast_allowed_avg", "games_sample"
    ]]

    # Limpiar tabla existente y guardar
    cursor = conn.cursor()
    cursor.execute("DELETE FROM defense_vs_position")
    conn.commit()

    dvp_stats_final.to_sql("defense_vs_position", conn, if_exists="append", index=False)

    conn.close()

    print(f"\nDvP calculado y guardado: {len(dvp_stats_final)} registros")

    return dvp_stats_final


def get_dvp_rating(team_abbrev: str, position: str, season: str = "2024-25"):
    """
    Obtiene el rating DvP para un equipo/posición específico.
    Retorna: (pts_allowed, reb_allowed, ast_allowed, rating)

    Rating: 1-5 donde 5 = muy mala defensa (bueno para el atacante)
    """
    conn = sqlite3.connect(DB_PATH)

    query = """
        SELECT pts_allowed_avg, reb_allowed_avg, ast_allowed_avg, games_sample
        FROM defense_vs_position
        WHERE team_abbrev = ? AND position = ? AND season = ?
    """

    result = pd.read_sql(query, conn, params=[team_abbrev, position, season])
    conn.close()

    if result.empty:
        return None

    row = result.iloc[0]

    # Calcular rating comparando con la media de la liga
    conn = sqlite3.connect(DB_PATH)
    league_avg = pd.read_sql("""
        SELECT
            AVG(pts_allowed_avg) as league_pts,
            STDEV(pts_allowed_avg) as std_pts
        FROM defense_vs_position
        WHERE position = ? AND season = ?
    """, conn, params=[position, season])
    conn.close()

    # Rating basado en desviaciones estándar
    if not league_avg.empty and league_avg["std_pts"].values[0]:
        z_score = (row["pts_allowed_avg"] - league_avg["league_pts"].values[0]) / league_avg["std_pts"].values[0]
        # Convertir z-score a rating 1-5
        rating = min(5, max(1, 3 + z_score))
    else:
        rating = 3  # Neutral

    return {
        "team": team_abbrev,
        "position": position,
        "pts_allowed": row["pts_allowed_avg"],
        "reb_allowed": row["reb_allowed_avg"],
        "ast_allowed": row["ast_allowed_avg"],
        "games_sample": row["games_sample"],
        "matchup_rating": round(rating, 1)  # 5 = mala defensa = BUENO para apostar Over
    }


def show_best_matchups(position: str = "C", stat: str = "pts", season: str = "2024-25"):
    """Muestra los mejores matchups para una posición."""
    conn = sqlite3.connect(DB_PATH)

    stat_col = f"{stat}_allowed_avg"

    query = f"""
        SELECT team_abbrev, position, {stat_col}, games_sample
        FROM defense_vs_position
        WHERE position = ? AND season = ?
        ORDER BY {stat_col} DESC
        LIMIT 10
    """

    result = pd.read_sql(query, conn, params=[position, season])
    conn.close()

    print(f"\n--- TOP 10 MATCHUPS PARA {position} ({stat.upper()}) ---")
    print(f"Equipos que MÁS {stat} permiten a la posición {position}:\n")
    print(result.to_string(index=False))

    return result


def show_dvp_summary():
    """Muestra un resumen del DvP calculado."""
    conn = sqlite3.connect(DB_PATH)

    # Por posición
    summary = pd.read_sql("""
        SELECT
            position,
            ROUND(AVG(pts_allowed_avg), 1) as avg_pts,
            ROUND(AVG(reb_allowed_avg), 1) as avg_reb,
            ROUND(AVG(ast_allowed_avg), 1) as avg_ast,
            SUM(games_sample) as total_games
        FROM defense_vs_position
        GROUP BY position
    """, conn)

    print("\n--- RESUMEN DvP POR POSICIÓN ---")
    print(summary.to_string(index=False))

    # Mejores y peores defensas vs Centers (ejemplo)
    print("\n--- DEFENSAS VS CENTERS (PTS) ---")

    best_def = pd.read_sql("""
        SELECT team_abbrev, pts_allowed_avg
        FROM defense_vs_position
        WHERE position = 'C'
        ORDER BY pts_allowed_avg ASC
        LIMIT 5
    """, conn)
    print("Mejores defensas (menos puntos permiten):")
    print(best_def.to_string(index=False))

    worst_def = pd.read_sql("""
        SELECT team_abbrev, pts_allowed_avg
        FROM defense_vs_position
        WHERE position = 'C'
        ORDER BY pts_allowed_avg DESC
        LIMIT 5
    """, conn)
    print("\nPeores defensas (TARGET para Overs):")
    print(worst_def.to_string(index=False))

    conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calcular Defense vs Position")
    parser.add_argument("--calculate", action="store_true", help="Calcular DvP desde game logs")
    parser.add_argument("--summary", action="store_true", help="Mostrar resumen DvP")
    parser.add_argument("--matchups", type=str, help="Mostrar mejores matchups para posición (PG,SG,SF,PF,C)")

    args = parser.parse_args()

    if args.calculate:
        calculate_dvp_from_game_logs()
        show_dvp_summary()
    elif args.summary:
        show_dvp_summary()
    elif args.matchups:
        show_best_matchups(position=args.matchups.upper())
    else:
        # Por defecto, calcular todo
        calculate_dvp_from_game_logs()
        show_dvp_summary()
