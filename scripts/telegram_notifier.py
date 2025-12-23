"""
Fase 5: Sistema de Notificaciones Telegram
Envía alertas cuando se detectan apuestas con valor.
"""

import os
import asyncio
from datetime import datetime
from typing import Optional

# Telegram Bot API
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


class TelegramNotifier:
    """Envía notificaciones a Telegram."""

    def __init__(self, bot_token: str = None, chat_id: str = None):
        self.bot_token = bot_token or TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or TELEGRAM_CHAT_ID

        if self.bot_token:
            try:
                from telegram import Bot
                self.bot = Bot(token=self.bot_token)
                self.enabled = True
            except ImportError:
                print("python-telegram-bot no instalado. Usando modo simulado.")
                self.bot = None
                self.enabled = False
        else:
            self.bot = None
            self.enabled = False

    async def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Envía un mensaje al chat configurado."""
        if not self.enabled or not self.chat_id:
            print(f"[TELEGRAM SIMULADO]\n{text}\n")
            return True

        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=parse_mode
            )
            return True
        except Exception as e:
            print(f"Error enviando mensaje: {e}")
            return False

    def send_sync(self, text: str, parse_mode: str = "HTML") -> bool:
        """Versión síncrona de send_message."""
        return asyncio.run(self.send_message(text, parse_mode))

    def format_value_bet_alert(self, bet: dict) -> str:
        """
        Formatea una alerta de apuesta con valor.

        Args:
            bet: Dict con información de la apuesta

        Returns:
            Mensaje formateado para Telegram
        """
        emoji_bet = "🟢" if bet.get("bet_type") == "OVER" else "🔴"
        emoji_conf = {
            "HIGH": "🔥",
            "MEDIUM": "⚡",
            "LOW": "💡"
        }.get(bet.get("confidence", "MEDIUM"), "⚡")

        message = f"""
{emoji_bet} <b>ALERTA DE VALOR</b> {emoji_conf}

<b>Jugador:</b> {bet.get('player', 'N/A')}
<b>Vs:</b> {bet.get('opponent', 'N/A')}
<b>Apuesta:</b> {bet.get('bet_type', 'N/A')} {bet.get('stat', '').upper()} {bet.get('line', 'N/A')}
<b>Cuota:</b> {bet.get('odds', 'N/A')}

📊 <b>Análisis:</b>
• Predicción modelo: {bet.get('model_prediction', 'N/A')}
• Prob modelo: {bet.get('model_probability', 'N/A')}%
• Prob mercado: {bet.get('implied_probability', 'N/A')}%
• <b>Edge: +{bet.get('edge', 'N/A')}%</b>
• <b>EV: +{bet.get('ev', 'N/A')}%</b>

🎯 <b>Confianza:</b> {bet.get('confidence', 'MEDIUM')}

{bet.get('reasoning', '')}

⏰ {datetime.now().strftime('%H:%M:%S')}
        """

        return message.strip()

    def format_daily_summary(self, bets: list, results: dict = None) -> str:
        """
        Formatea el resumen diario.

        Args:
            bets: Lista de apuestas del día
            results: Resultados de apuestas anteriores

        Returns:
            Mensaje formateado
        """
        total_bets = len(bets)
        total_ev = sum(b.get("ev", 0) for b in bets)
        avg_ev = total_ev / total_bets if total_bets > 0 else 0

        by_type = {"OVER": 0, "UNDER": 0, "NO_BET": 0}
        for bet in bets:
            bet_type = bet.get("bet_type", "NO_BET")
            by_type[bet_type] = by_type.get(bet_type, 0) + 1

        message = f"""
📊 <b>RESUMEN DIARIO NBA PROPS</b>

📅 {datetime.now().strftime('%Y-%m-%d')}

<b>Apuestas analizadas:</b> {total_bets}
<b>EV promedio:</b> {avg_ev:.1f}%

<b>Distribución:</b>
• OVER: {by_type['OVER']}
• UNDER: {by_type['UNDER']}
• NO BET: {by_type['NO_BET']}
        """

        if results:
            wins = results.get("wins", 0)
            losses = results.get("losses", 0)
            total = wins + losses
            win_rate = (wins / total * 100) if total > 0 else 0

            message += f"""

<b>Resultados del día anterior:</b>
✅ Ganadas: {wins}
❌ Perdidas: {losses}
📈 Win rate: {win_rate:.1f}%
        """

        return message.strip()

    def format_agent_analysis(self, analysis: dict) -> str:
        """
        Formatea el análisis completo del sistema de agentes.

        Args:
            analysis: Output del NBABettingAgentSystem.analyze_player_prop()

        Returns:
            Mensaje formateado
        """
        player = analysis.get("player", "N/A")
        stat = analysis.get("stat", "").upper()
        line = analysis.get("line", "N/A")
        odds = analysis.get("odds", "N/A")

        # Extraer la decisión final del manager agent
        final_decision = analysis.get("final_decision", "")

        # Determinar el tipo de apuesta del texto
        bet_type = "NO_BET"
        if "OVER" in final_decision.upper():
            bet_type = "OVER"
            emoji = "🟢"
        elif "UNDER" in final_decision.upper():
            bet_type = "UNDER"
            emoji = "🔴"
        else:
            emoji = "⚪"

        message = f"""
{emoji} <b>ANÁLISIS COMPLETO</b>

<b>Jugador:</b> {player}
<b>Prop:</b> {stat} O/U {line} @ {odds}

📰 <b>News Agent:</b>
{self._truncate(analysis.get('news_analysis', 'N/A'), 200)}

📋 <b>Lineup Agent:</b>
{self._truncate(analysis.get('lineup_analysis', 'N/A'), 200)}

🎯 <b>Decisión Final:</b>
{self._truncate(final_decision, 300)}

⏰ {datetime.now().strftime('%H:%M:%S')}
        """

        return message.strip()

    def _truncate(self, text: str, max_len: int) -> str:
        """Trunca texto si es muy largo."""
        if len(text) <= max_len:
            return text
        return text[:max_len-3] + "..."


class AlertManager:
    """Gestiona las alertas y evita spam."""

    def __init__(self, notifier: TelegramNotifier, cooldown_minutes: int = 30):
        self.notifier = notifier
        self.cooldown = cooldown_minutes * 60  # En segundos
        self.sent_alerts = {}  # {alert_key: timestamp}

    def should_send_alert(self, alert_key: str) -> bool:
        """Verifica si se debe enviar una alerta (evita duplicados)."""
        now = datetime.now().timestamp()

        if alert_key in self.sent_alerts:
            last_sent = self.sent_alerts[alert_key]
            if now - last_sent < self.cooldown:
                return False

        return True

    def send_value_bet_alert(self, bet: dict) -> bool:
        """Envía alerta de apuesta con valor si no se envió recientemente."""
        alert_key = f"{bet.get('player')}_{bet.get('stat')}_{bet.get('bet_type')}"

        if not self.should_send_alert(alert_key):
            print(f"Alerta {alert_key} en cooldown, no se envía.")
            return False

        message = self.notifier.format_value_bet_alert(bet)
        success = self.notifier.send_sync(message)

        if success:
            self.sent_alerts[alert_key] = datetime.now().timestamp()

        return success

    def send_batch_alerts(self, bets: list, max_alerts: int = 5) -> int:
        """Envía múltiples alertas (limitadas)."""
        sent_count = 0

        for bet in bets[:max_alerts]:
            if self.send_value_bet_alert(bet):
                sent_count += 1

        return sent_count


def run_alert_system():
    """
    Sistema principal de alertas.
    Escanea el mercado y envía alertas de valor.
    """
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))

    from odds_scraper import scan_for_value

    print("="*60)
    print("SISTEMA DE ALERTAS NBA PROPS")
    print("="*60)

    # Inicializar notificador
    notifier = TelegramNotifier()
    alert_manager = AlertManager(notifier, cooldown_minutes=30)

    # Escanear mercado
    print("\nEscaneando mercado...")
    value_bets = scan_for_value()

    if not value_bets:
        print("\nNo hay apuestas con valor para alertar.")
        return

    # Filtrar solo las de alto valor (EV > 8%)
    high_value = [b for b in value_bets if b.get("ev", 0) > 8]

    print(f"\nApuestas de alto valor encontradas: {len(high_value)}")

    # Enviar alertas
    sent = alert_manager.send_batch_alerts(high_value, max_alerts=3)
    print(f"Alertas enviadas: {sent}")

    # Enviar resumen diario
    if datetime.now().hour == 10:  # A las 10am
        summary = notifier.format_daily_summary(value_bets)
        notifier.send_sync(summary)


def demo():
    """Demo del sistema de notificaciones."""
    notifier = TelegramNotifier()

    # Simular una apuesta con valor
    sample_bet = {
        "player": "LeBron James",
        "opponent": "GSW",
        "stat": "pts",
        "bet_type": "OVER",
        "line": 25.5,
        "odds": 1.91,
        "model_prediction": 27.3,
        "model_probability": 58.5,
        "implied_probability": 52.4,
        "edge": 6.1,
        "ev": 11.8,
        "confidence": "HIGH",
        "reasoning": "Tendencia positiva últimos 5 partidos. GSW permite muchos puntos a SFs."
    }

    print("Enviando alerta de prueba...")
    message = notifier.format_value_bet_alert(sample_bet)
    notifier.send_sync(message)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sistema de alertas Telegram")
    parser.add_argument("--run", action="store_true", help="Ejecutar sistema de alertas")
    parser.add_argument("--demo", action="store_true", help="Enviar mensaje de prueba")

    args = parser.parse_args()

    if args.run:
        run_alert_system()
    elif args.demo:
        demo()
    else:
        print("Usa --run para ejecutar o --demo para probar")
