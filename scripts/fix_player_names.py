"""
Script para arreglar los nombres de jugadores en la base de datos.
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "nba_props.db"

def fix_names():
    print("Arreglando nombres de jugadores...")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Actualizar nombres usando la tabla players
    cursor.execute("""
        UPDATE player_game_logs
        SET player_name = (
            SELECT player_name FROM players
            WHERE players.player_id = player_game_logs.player_id
        )
        WHERE player_name IS NULL
    """)

    updated = cursor.rowcount
    conn.commit()

    print(f"Actualizados {updated} registros")

    # Verificar
    cursor.execute("SELECT player_name, pts, reb, ast FROM player_game_logs WHERE player_name IS NOT NULL LIMIT 10")
    print("\nMuestra de datos corregidos:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]} pts, {row[2]} reb, {row[3]} ast")

    # Contar cuántos siguen sin nombre
    cursor.execute("SELECT COUNT(*) FROM player_game_logs WHERE player_name IS NULL")
    still_null = cursor.fetchone()[0]
    print(f"\nRegistros aún sin nombre: {still_null}")

    conn.close()

if __name__ == "__main__":
    fix_names()
