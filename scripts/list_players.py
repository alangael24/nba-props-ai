"""Lista jugadores disponibles en la base de datos."""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "nba_props.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Top jugadores por puntos
print("=== TOP 20 JUGADORES POR PPG ===\n")
cursor.execute("""
    SELECT player_name,
           ROUND(AVG(pts), 1) as ppg,
           ROUND(AVG(reb), 1) as rpg,
           ROUND(AVG(ast), 1) as apg,
           COUNT(*) as games
    FROM player_game_logs
    WHERE player_name IS NOT NULL AND min > 20
    GROUP BY player_name
    HAVING games > 10
    ORDER BY ppg DESC
    LIMIT 20
""")

for row in cursor.fetchall():
    print(f"{row[0]}: {row[1]} pts, {row[2]} reb, {row[3]} ast ({row[4]} games)")

# Buscar jugadores específicos
print("\n=== BUSCAR JUGADOR ===")
search = input("Nombre a buscar (o Enter para saltar): ")
if search:
    cursor.execute("""
        SELECT DISTINCT player_name FROM player_game_logs
        WHERE player_name LIKE ?
    """, (f"%{search}%",))
    results = cursor.fetchall()
    if results:
        for r in results:
            print(f"  - {r[0]}")
    else:
        print("  No encontrado")

conn.close()
