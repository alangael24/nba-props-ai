#!/usr/bin/env python3
"""
NBA PROPS AI - Sistema de Predicción con Agentes Claude

Sistema completo para predecir Player Props de la NBA usando:
- Datos históricos de la NBA API
- Modelo XGBoost para predicciones
- Agentes Claude para análisis de noticias y lineups
- Sistema de alertas Telegram

Uso:
    python main.py --setup           # Primera vez: descargar datos
    python main.py --train           # Entrenar modelos
    python main.py --scan            # Escanear mercado
    python main.py --analyze PLAYER  # Análisis completo de un jugador
    python main.py --alerts          # Activar sistema de alertas
"""

import argparse
import sys
from pathlib import Path

# Agregar paths
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent / "models"))
sys.path.insert(0, str(Path(__file__).parent / "agents"))


def setup_database():
    """Fase 1: Configurar base de datos e ingestar datos históricos."""
    print("\n" + "="*60)
    print("FASE 1: SETUP DE BASE DE DATOS")
    print("="*60)

    from ingest_nba_data import run_full_ingestion
    run_full_ingestion()

    print("\n¿Deseas calcular Defense vs Position? (y/n): ", end="")
    if input().lower() == 'y':
        from calculate_dvp import calculate_dvp_from_game_logs, show_dvp_summary
        calculate_dvp_from_game_logs()
        show_dvp_summary()


def train_models():
    """Fase 2: Entrenar modelos XGBoost."""
    print("\n" + "="*60)
    print("FASE 2: ENTRENAMIENTO DE MODELOS")
    print("="*60)

    from xgboost_predictor import NBAPropsPredictor

    predictor = NBAPropsPredictor()
    predictor.train()


def scan_market(min_ev: float = 0.05):
    """Fase 4: Escanear mercado buscando valor."""
    print("\n" + "="*60)
    print("FASE 4: ESCÁNER DE MERCADO")
    print("="*60)

    from odds_scraper import scan_for_value
    return scan_for_value()


def analyze_player(player_name: str, opponent: str, stat: str = "pts",
                   line: float = 25.5, odds: float = 1.90, is_home: bool = True):
    """Fase 3+5: Análisis completo usando agentes Claude."""
    print("\n" + "="*60)
    print("ANÁLISIS COMPLETO CON AGENTES CLAUDE")
    print("="*60)

    from nba_agents import NBABettingAgentSystem

    system = NBABettingAgentSystem()
    result = system.analyze_player_prop(
        player_name=player_name,
        opponent=opponent,
        stat=stat,
        line=line,
        odds=odds,
        is_home=is_home
    )

    return result


def run_alerts():
    """Fase 5: Activar sistema de alertas."""
    print("\n" + "="*60)
    print("SISTEMA DE ALERTAS ACTIVADO")
    print("="*60)

    from telegram_notifier import run_alert_system
    run_alert_system()


def quick_predict(player_name: str, opponent: str):
    """Predicción rápida sin agentes (solo modelo XGBoost)."""
    from xgboost_predictor import NBAPropsPredictor

    predictor = NBAPropsPredictor()
    result = predictor.predict_player(player_name, opponent)

    print("\n" + "="*60)
    print(f"PREDICCIÓN RÁPIDA: {player_name} vs {opponent}")
    print("="*60)

    if "error" in result:
        print(f"Error: {result['error']}")
        return result

    print(f"\nPredicciones:")
    print(f"  Puntos: {result['predictions']['pts']}")
    print(f"  Rebotes: {result['predictions']['reb']}")
    print(f"  Asistencias: {result['predictions']['ast']}")

    print(f"\nVolatilidad (std):")
    print(f"  Puntos: ±{result['std']['pts']}")
    print(f"  Rebotes: ±{result['std']['reb']}")
    print(f"  Asistencias: ±{result['std']['ast']}")

    print(f"\nHistórico:")
    print(f"  Promedio últimos 5: {result['historical']['avg_pts_5']}")
    print(f"  Promedio temporada: {result['historical']['avg_pts_season']}")

    return result


def show_status():
    """Muestra el estado del sistema."""
    import sqlite3
    from pathlib import Path

    db_path = Path(__file__).parent / "data" / "nba_props.db"

    print("\n" + "="*60)
    print("ESTADO DEL SISTEMA NBA PROPS AI")
    print("="*60)

    # Base de datos
    if db_path.exists():
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM player_game_logs")
        total_games = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT player_id) FROM player_game_logs")
        total_players = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM defense_vs_position")
        dvp_records = cursor.fetchone()[0]

        conn.close()

        print(f"\n📊 Base de datos: ACTIVA")
        print(f"   - Game logs: {total_games:,}")
        print(f"   - Jugadores: {total_players:,}")
        print(f"   - DvP records: {dvp_records:,}")
    else:
        print(f"\n⚠️  Base de datos: NO ENCONTRADA")
        print(f"   Ejecuta: python main.py --setup")

    # Modelos
    models_path = Path(__file__).parent / "models"
    model_pts = models_path / "model_pts.joblib"

    if model_pts.exists():
        print(f"\n🤖 Modelos XGBoost: ENTRENADOS")
    else:
        print(f"\n⚠️  Modelos XGBoost: NO ENTRENADOS")
        print(f"   Ejecuta: python main.py --train")

    # API Keys
    import os

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    odds_api_key = os.environ.get("ODDS_API_KEY", "")

    print(f"\n🔑 API Keys:")
    print(f"   - ANTHROPIC_API_KEY: {'✅ Configurada' if anthropic_key else '❌ No configurada'}")
    print(f"   - TELEGRAM_BOT_TOKEN: {'✅ Configurado' if telegram_token else '❌ No configurado'}")
    print(f"   - ODDS_API_KEY: {'✅ Configurada' if odds_api_key else '❌ No configurada (usando mock)'}")


def main():
    parser = argparse.ArgumentParser(
        description="NBA Props AI - Sistema de Predicción",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python main.py --setup                         # Setup inicial
  python main.py --train                         # Entrenar modelos
  python main.py --scan                          # Escanear mercado
  python main.py --predict "LeBron James" GSW    # Predicción rápida
  python main.py --analyze "LeBron James" GSW --stat pts --line 25.5
        """
    )

    parser.add_argument("--setup", action="store_true",
                        help="Setup inicial: descargar datos NBA")
    parser.add_argument("--train", action="store_true",
                        help="Entrenar modelos XGBoost")
    parser.add_argument("--scan", action="store_true",
                        help="Escanear mercado buscando valor")
    parser.add_argument("--alerts", action="store_true",
                        help="Activar sistema de alertas Telegram")
    parser.add_argument("--status", action="store_true",
                        help="Mostrar estado del sistema")
    parser.add_argument("--backtest", action="store_true",
                        help="Ejecutar backtest walk-forward")

    parser.add_argument("--predict", nargs=2, metavar=("PLAYER", "OPPONENT"),
                        help="Predicción rápida (solo XGBoost)")
    parser.add_argument("--analyze", nargs=2, metavar=("PLAYER", "OPPONENT"),
                        help="Análisis completo con agentes Claude")

    parser.add_argument("--stat", type=str, default="pts",
                        choices=["pts", "reb", "ast"],
                        help="Estadística a analizar")
    parser.add_argument("--line", type=float, default=25.5,
                        help="Línea de apuesta")
    parser.add_argument("--odds", type=float, default=1.90,
                        help="Cuota decimal")
    parser.add_argument("--home", action="store_true",
                        help="El jugador juega de local")

    args = parser.parse_args()

    # Ejecutar acción correspondiente
    if args.setup:
        setup_database()

    elif args.train:
        train_models()

    elif args.scan:
        scan_market()

    elif args.alerts:
        run_alerts()

    elif args.status:
        show_status()

    elif args.backtest:
        from backtester import WalkForwardBacktester
        backtester = WalkForwardBacktester(
            initial_bankroll=1000.0,
            bet_size=50.0,
            min_ev_threshold=0.05
        )
        backtester.run_backtest(
            start_date="2023-10-01",
            end_date="2024-04-01",
            retrain_frequency_days=14
        )
        backtester.print_results()

    elif args.predict:
        player, opponent = args.predict
        quick_predict(player, opponent)

    elif args.analyze:
        player, opponent = args.analyze
        analyze_player(
            player_name=player,
            opponent=opponent,
            stat=args.stat,
            line=args.line,
            odds=args.odds,
            is_home=args.home
        )

    else:
        # Mostrar estado por defecto
        show_status()
        print("\nUsa --help para ver opciones disponibles")


if __name__ == "__main__":
    main()
