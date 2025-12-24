"""
Fase 3: Red de Agentes con Claude API
Sistema de agentes colaborativos para predicción de NBA Props.

Agentes:
1. News Agent - Busca noticias y estado de jugadores
2. Lineup Agent - Detecta bajas y cambios en roster
3. Manager Agent - Toma la decisión final de apuesta
"""

import os
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import anthropic
import httpx

# API Keys para búsqueda de noticias
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")  # Serper.dev - $50 por 50K búsquedas

# Paths
DB_PATH = Path(__file__).parent.parent / "data" / "nba_props.db"

# Cliente Anthropic
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# =============================================================================
# HERRAMIENTAS (Tools) - Definiciones JSON Schema
# =============================================================================

SEARCH_NEWS_TOOL = {
    "name": "search_player_news",
    "description": "Busca noticias recientes sobre un jugador de NBA. Retorna estado de lesiones, restricciones de minutos, y cualquier información que afecte su rendimiento.",
    "input_schema": {
        "type": "object",
        "properties": {
            "player_name": {
                "type": "string",
                "description": "Nombre completo del jugador (ej: LeBron James)"
            },
            "search_keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Palabras clave adicionales (ej: injury, rest, DNP)"
            }
        },
        "required": ["player_name"]
    }
}

CHECK_LINEUP_TOOL = {
    "name": "check_team_lineup",
    "description": "Verifica el estado del lineup de un equipo. Detecta jugadores OUT, DOUBTFUL, QUESTIONABLE y cambios recientes.",
    "input_schema": {
        "type": "object",
        "properties": {
            "team_abbrev": {
                "type": "string",
                "description": "Abreviatura del equipo (ej: LAL, BOS, GSW)"
            }
        },
        "required": ["team_abbrev"]
    }
}

GET_PLAYER_STATS_TOOL = {
    "name": "get_player_historical_stats",
    "description": "Obtiene estadísticas históricas de un jugador desde la base de datos local.",
    "input_schema": {
        "type": "object",
        "properties": {
            "player_name": {
                "type": "string",
                "description": "Nombre del jugador"
            },
            "last_n_games": {
                "type": "integer",
                "description": "Número de partidos recientes a analizar",
                "default": 10
            },
            "vs_opponent": {
                "type": "string",
                "description": "Filtrar por oponente específico (abreviatura)"
            }
        },
        "required": ["player_name"]
    }
}

EVALUATE_BET_TOOL = {
    "name": "evaluate_betting_value",
    "description": "Evalúa si una apuesta tiene Expected Value positivo usando el modelo XGBoost.",
    "input_schema": {
        "type": "object",
        "properties": {
            "player_name": {
                "type": "string",
                "description": "Nombre del jugador"
            },
            "opponent": {
                "type": "string",
                "description": "Equipo rival (abreviatura)"
            },
            "stat": {
                "type": "string",
                "enum": ["pts", "reb", "ast"],
                "description": "Estadística a predecir"
            },
            "line": {
                "type": "number",
                "description": "Línea de la apuesta (ej: 25.5)"
            },
            "odds": {
                "type": "number",
                "description": "Cuota decimal (ej: 1.90)"
            },
            "is_home": {
                "type": "boolean",
                "description": "Si el jugador juega de local"
            }
        },
        "required": ["player_name", "opponent", "stat", "line", "odds"]
    }
}

MAKE_FINAL_DECISION_TOOL = {
    "name": "make_final_decision",
    "description": "Registra la decisión final de apuesta con justificación completa.",
    "input_schema": {
        "type": "object",
        "properties": {
            "player_name": {"type": "string"},
            "bet_type": {
                "type": "string",
                "enum": ["OVER", "UNDER", "NO_BET"],
                "description": "Tipo de apuesta recomendada"
            },
            "stat": {"type": "string"},
            "line": {"type": "number"},
            "confidence": {
                "type": "string",
                "enum": ["HIGH", "MEDIUM", "LOW"],
                "description": "Nivel de confianza"
            },
            "reasoning": {
                "type": "string",
                "description": "Justificación detallada de la decisión"
            },
            "risk_factors": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Factores de riesgo identificados"
            }
        },
        "required": ["player_name", "bet_type", "stat", "confidence", "reasoning"]
    }
}


# =============================================================================
# IMPLEMENTACIÓN DE HERRAMIENTAS
# =============================================================================

def _search_serper(query: str, num_results: int = 5) -> list:
    """
    Busca en Google usando Serper.dev API.

    Returns:
        Lista de resultados con title, snippet, link
    """
    if not SERPER_API_KEY:
        return []

    try:
        response = httpx.post(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY": SERPER_API_KEY,
                "Content-Type": "application/json"
            },
            json={
                "q": query,
                "num": num_results,
                "gl": "us",
                "hl": "en"
            },
            timeout=10.0
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("organic", [])[:num_results]:
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "link": item.get("link", ""),
                "date": item.get("date", "")
            })

        return results
    except Exception as e:
        print(f"  [ERROR] Serper API: {e}")
        return []


def _analyze_news_for_risk(news_items: list, player_name: str) -> dict:
    """
    Analiza noticias para detectar lesiones, restricciones, etc.

    Returns:
        Dict con status, injury_report, minutes_restriction, risk_level
    """
    # Keywords para detectar problemas
    injury_keywords = ["injury", "injured", "hurt", "pain", "sore", "sprain", "strain",
                       "out", "miss", "sideline", "surgery", "rehab"]
    questionable_keywords = ["questionable", "doubtful", "game-time", "uncertain", "day-to-day"]
    rest_keywords = ["rest", "load management", "dnp", "sit out", "night off"]
    restriction_keywords = ["minutes restriction", "limited minutes", "minute limit",
                           "pitch count", "ramp up"]
    positive_keywords = ["cleared", "full practice", "no restrictions", "healthy",
                        "ready to play", "available", "probable"]

    all_text = " ".join([
        f"{item.get('title', '')} {item.get('snippet', '')}".lower()
        for item in news_items
    ])

    # Detectar status
    status = "ACTIVE"
    injury_report = None
    minutes_restriction = None
    risk_level = "LOW"

    # Buscar indicadores negativos
    has_injury = any(kw in all_text for kw in injury_keywords)
    has_questionable = any(kw in all_text for kw in questionable_keywords)
    has_rest = any(kw in all_text for kw in rest_keywords)
    has_restriction = any(kw in all_text for kw in restriction_keywords)
    has_positive = any(kw in all_text for kw in positive_keywords)

    if "out" in all_text and player_name.lower().split()[0] in all_text:
        status = "OUT"
        risk_level = "CRITICAL"
    elif has_questionable:
        status = "QUESTIONABLE"
        risk_level = "HIGH"
    elif has_injury or has_rest:
        status = "PROBABLE"
        risk_level = "MEDIUM"
    elif has_positive:
        status = "ACTIVE"
        risk_level = "LOW"

    # Extraer detalles de lesión
    if has_injury:
        for item in news_items:
            snippet = item.get("snippet", "").lower()
            if any(kw in snippet for kw in injury_keywords):
                injury_report = item.get("snippet", "")[:200]
                break

    # Extraer restricción de minutos
    if has_restriction:
        for item in news_items:
            snippet = item.get("snippet", "").lower()
            if any(kw in snippet for kw in restriction_keywords):
                minutes_restriction = item.get("snippet", "")[:150]
                break

    return {
        "status": status,
        "injury_report": injury_report,
        "minutes_restriction": minutes_restriction,
        "risk_level": risk_level
    }


def execute_search_news(player_name: str, search_keywords: list = None) -> dict:
    """
    Busca noticias sobre un jugador usando Serper.dev API.

    Si SERPER_API_KEY no está configurada, usa datos simulados como fallback.
    """
    print(f"  [TOOL] Buscando noticias de: {player_name}")

    # Construir query de búsqueda
    base_query = f"{player_name} NBA"
    if search_keywords:
        base_query += " " + " ".join(search_keywords)
    else:
        base_query += " injury status news today"

    # Intentar búsqueda real con Serper
    if SERPER_API_KEY:
        print(f"  [SERPER] Query: {base_query}")
        news_items = _search_serper(base_query, num_results=5)

        if news_items:
            # Analizar noticias para extraer info relevante
            analysis = _analyze_news_for_risk(news_items, player_name)

            return {
                "player": player_name,
                "status": analysis["status"],
                "injury_report": analysis["injury_report"],
                "minutes_restriction": analysis["minutes_restriction"],
                "last_update": datetime.now().strftime("%Y-%m-%d"),
                "recent_news": [item.get("snippet", "")[:200] for item in news_items[:3]],
                "risk_level": analysis["risk_level"],
                "source": "serper_api",
                "raw_results": news_items
            }
        else:
            print("  [SERPER] No results, falling back to simulated data")

    # Fallback: datos simulados
    print("  [FALLBACK] Usando datos simulados")
    simulated_news = {
        "LeBron James": {
            "status": "ACTIVE",
            "injury_report": None,
            "minutes_restriction": None,
            "last_update": datetime.now().strftime("%Y-%m-%d"),
            "recent_news": [
                "LeBron participó en entrenamiento completo",
                "Sin restricciones de minutos reportadas"
            ],
            "risk_level": "LOW"
        },
        "Stephen Curry": {
            "status": "QUESTIONABLE",
            "injury_report": "Dolor en tobillo derecho",
            "minutes_restriction": "Posible límite de 28-30 minutos",
            "last_update": datetime.now().strftime("%Y-%m-%d"),
            "recent_news": [
                "Curry listado como QUESTIONABLE para el partido de hoy",
                "Se evaluará en calentamiento"
            ],
            "risk_level": "HIGH"
        },
        "Giannis Antetokounmpo": {
            "status": "ACTIVE",
            "injury_report": None,
            "minutes_restriction": None,
            "last_update": datetime.now().strftime("%Y-%m-%d"),
            "recent_news": [
                "Giannis confirmado como titular",
                "Regresa tras descanso programado"
            ],
            "risk_level": "LOW"
        }
    }

    default_response = {
        "status": "UNKNOWN",
        "injury_report": "No hay información disponible",
        "minutes_restriction": None,
        "last_update": datetime.now().strftime("%Y-%m-%d"),
        "recent_news": ["Sin noticias recientes encontradas"],
        "risk_level": "MEDIUM",
        "source": "simulated"
    }

    result = simulated_news.get(player_name, default_response)
    result["player"] = player_name
    result["source"] = "simulated"

    return result


TEAM_FULL_NAMES = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards"
}


def _parse_injury_report(news_items: list, team_name: str) -> dict:
    """
    Parsea noticias de lesiones para extraer jugadores OUT/DOUBTFUL/QUESTIONABLE.
    """
    out = []
    doubtful = []
    questionable = []
    probable = []
    key_player_status = {}

    for item in news_items:
        text = f"{item.get('title', '')} {item.get('snippet', '')}".lower()

        # Buscar patrones de lesión
        if "out" in text:
            # Intentar extraer nombres de jugadores
            words = text.split()
            for i, word in enumerate(words):
                if word == "out" and i > 0:
                    # Nombre probable antes de "out"
                    potential_name = " ".join(words[max(0,i-2):i]).title()
                    if len(potential_name) > 3 and potential_name not in out:
                        out.append(potential_name.strip(".,"))

        if "questionable" in text or "game-time decision" in text:
            words = text.split()
            for i, word in enumerate(words):
                if "questionable" in word and i > 0:
                    potential_name = " ".join(words[max(0,i-2):i]).title()
                    if len(potential_name) > 3 and potential_name not in questionable:
                        questionable.append(potential_name.strip(".,"))

        if "doubtful" in text:
            words = text.split()
            for i, word in enumerate(words):
                if word == "doubtful" and i > 0:
                    potential_name = " ".join(words[max(0,i-2):i]).title()
                    if len(potential_name) > 3 and potential_name not in doubtful:
                        doubtful.append(potential_name.strip(".,"))

    return {
        "out": out[:5],  # Limitar a 5
        "doubtful": doubtful[:3],
        "questionable": questionable[:5],
        "probable": probable[:3],
        "key_player_status": key_player_status
    }


def execute_check_lineup(team_abbrev: str) -> dict:
    """
    Verifica el estado del lineup de un equipo usando Serper API.

    Busca injury reports en fuentes como Rotowire, ESPN, CBS Sports.
    """
    print(f"  [TOOL] Verificando lineup de: {team_abbrev}")

    team_name = TEAM_FULL_NAMES.get(team_abbrev, team_abbrev)

    # Intentar búsqueda real con Serper
    if SERPER_API_KEY:
        query = f"{team_name} injury report today NBA"
        print(f"  [SERPER] Query: {query}")
        news_items = _search_serper(query, num_results=5)

        if news_items:
            # Parsear injury report
            injury_data = _parse_injury_report(news_items, team_name)

            # Determinar impacto
            usage_impact = "Información obtenida de fuentes reales"
            if injury_data["out"]:
                usage_impact = f"Jugadores OUT: {', '.join(injury_data['out'][:3])}"

            return {
                "team": team_name,
                "team_abbrev": team_abbrev,
                "out": injury_data["out"],
                "doubtful": injury_data["doubtful"],
                "questionable": injury_data["questionable"],
                "probable": injury_data["probable"],
                "key_player_status": injury_data["key_player_status"],
                "usage_impact": usage_impact,
                "source": "serper_api",
                "raw_news": [item.get("snippet", "")[:150] for item in news_items[:3]]
            }
        else:
            print("  [SERPER] No results, falling back to simulated data")

    # Fallback: datos simulados
    print("  [FALLBACK] Usando datos simulados")
    lineups = {
        "LAL": {
            "team": "Los Angeles Lakers",
            "out": ["Jarred Vanderbilt"],
            "doubtful": [],
            "questionable": ["Austin Reaves"],
            "probable": ["LeBron James"],
            "key_player_status": {
                "LeBron James": "PROBABLE - Sin restricciones",
                "Anthony Davis": "AVAILABLE"
            },
            "usage_impact": "Si Reaves está OUT, esperar +3% usage para LeBron"
        },
        "GSW": {
            "team": "Golden State Warriors",
            "out": [],
            "doubtful": [],
            "questionable": ["Stephen Curry"],
            "probable": ["Draymond Green"],
            "key_player_status": {
                "Stephen Curry": "QUESTIONABLE - Tobillo",
                "Klay Thompson": "AVAILABLE"
            },
            "usage_impact": "Si Curry está OUT, +15% usage para Kuminga y Wiggins"
        },
        "BOS": {
            "team": "Boston Celtics",
            "out": [],
            "doubtful": [],
            "questionable": [],
            "probable": [],
            "key_player_status": {
                "Jayson Tatum": "AVAILABLE",
                "Jaylen Brown": "AVAILABLE"
            },
            "usage_impact": "Lineup completo esperado"
        }
    }

    result = lineups.get(team_abbrev, {
        "team": team_name,
        "team_abbrev": team_abbrev,
        "out": [],
        "doubtful": [],
        "questionable": [],
        "probable": [],
        "key_player_status": {},
        "usage_impact": "Información no disponible",
        "source": "simulated"
    })
    result["source"] = "simulated"

    return result


def execute_get_player_stats(player_name: str, last_n_games: int = 10,
                              vs_opponent: str = None) -> dict:
    """Obtiene estadísticas históricas desde la base de datos."""
    print(f"  [TOOL] Obteniendo stats de: {player_name} (últimos {last_n_games} partidos)")

    try:
        conn = sqlite3.connect(DB_PATH)

        # Query base
        query = """
            SELECT
                game_date, matchup, pts, reb, ast, min,
                is_home, opponent_abbrev
            FROM player_game_logs
            WHERE player_name = ?
            ORDER BY game_date DESC
            LIMIT ?
        """

        import pandas as pd
        df = pd.read_sql(query, conn, params=[player_name, last_n_games])

        if df.empty:
            conn.close()
            return {"error": f"No se encontró al jugador: {player_name}"}

        # Stats generales
        result = {
            "player": player_name,
            "games_analyzed": len(df),
            "averages": {
                "pts": round(df["pts"].mean(), 1),
                "reb": round(df["reb"].mean(), 1),
                "ast": round(df["ast"].mean(), 1),
                "min": round(df["min"].mean(), 1)
            },
            "std_dev": {
                "pts": round(df["pts"].std(), 2),
                "reb": round(df["reb"].std(), 2),
                "ast": round(df["ast"].std(), 2)
            },
            "last_5_games": df.head(5)[["game_date", "matchup", "pts", "reb", "ast"]].to_dict("records"),
            "trend": {
                "pts": round(df.head(5)["pts"].mean() - df["pts"].mean(), 1),
                "reb": round(df.head(5)["reb"].mean() - df["reb"].mean(), 1),
                "ast": round(df.head(5)["ast"].mean() - df["ast"].mean(), 1)
            }
        }

        # Stats vs oponente específico
        if vs_opponent:
            vs_opp = df[df["opponent_abbrev"] == vs_opponent]
            if not vs_opp.empty:
                result["vs_opponent"] = {
                    "opponent": vs_opponent,
                    "games": len(vs_opp),
                    "avg_pts": round(vs_opp["pts"].mean(), 1),
                    "avg_reb": round(vs_opp["reb"].mean(), 1),
                    "avg_ast": round(vs_opp["ast"].mean(), 1)
                }

        conn.close()
        return result

    except Exception as e:
        return {"error": str(e)}


def execute_evaluate_bet(player_name: str, opponent: str, stat: str,
                         line: float, odds: float, is_home: bool = True) -> dict:
    """Evalúa una apuesta usando el modelo XGBoost."""
    print(f"  [TOOL] Evaluando apuesta: {player_name} {stat} O/U {line} @ {odds}")

    try:
        # Importar el predictor
        import sys
        sys.path.append(str(Path(__file__).parent.parent / "models"))
        from xgboost_predictor import NBAPropsPredictor

        predictor = NBAPropsPredictor()
        result = predictor.evaluate_bet(
            player_name=player_name,
            opponent=opponent,
            stat=stat,
            line=line,
            odds=odds,
            is_home=is_home
        )

        return result

    except Exception as e:
        # Fallback si el modelo no está entrenado
        return {
            "player": player_name,
            "stat": stat,
            "line": line,
            "odds": odds,
            "model_status": "NOT_TRAINED",
            "message": f"Modelo no disponible: {str(e)}. Ejecuta --train primero."
        }


def execute_make_decision(player_name: str, bet_type: str, stat: str,
                          line: float = None, confidence: str = "MEDIUM",
                          reasoning: str = "", risk_factors: list = None) -> dict:
    """Registra la decisión final de apuesta."""
    print(f"  [TOOL] Decisión final: {player_name} - {bet_type} {stat}")

    decision = {
        "timestamp": datetime.now().isoformat(),
        "player": player_name,
        "decision": bet_type,
        "stat": stat,
        "line": line,
        "confidence": confidence,
        "reasoning": reasoning,
        "risk_factors": risk_factors or [],
        "status": "RECORDED"
    }

    # En producción: guardar en DB y/o enviar notificación
    return decision


def process_tool_call(tool_name: str, tool_input: dict) -> Any:
    """Router de herramientas."""
    tools_map = {
        "search_player_news": execute_search_news,
        "check_team_lineup": execute_check_lineup,
        "get_player_historical_stats": execute_get_player_stats,
        "evaluate_betting_value": execute_evaluate_bet,
        "make_final_decision": execute_make_decision
    }

    if tool_name in tools_map:
        return tools_map[tool_name](**tool_input)
    else:
        return {"error": f"Herramienta no encontrada: {tool_name}"}


# =============================================================================
# SISTEMA DE AGENTES
# =============================================================================

class NBABettingAgentSystem:
    """Sistema de agentes colaborativos para predicción de NBA Props."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model
        self.conversation_history = []

    def run_agent(self, agent_name: str, system_prompt: str,
                  user_prompt: str, tools: list, max_turns: int = 10) -> str:
        """
        Ejecuta un agente con su propio contexto.

        Args:
            agent_name: Nombre identificador del agente
            system_prompt: Instrucciones del sistema para el agente
            user_prompt: Tarea específica a realizar
            tools: Herramientas disponibles para el agente
            max_turns: Máximo de iteraciones

        Returns:
            Respuesta final del agente
        """
        print(f"\n{'='*60}")
        print(f"AGENTE: {agent_name}")
        print(f"{'='*60}")

        messages = [{"role": "user", "content": user_prompt}]

        for turn in range(max_turns):
            print(f"\n[Turn {turn + 1}]")

            response = client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=system_prompt,
                tools=tools,
                messages=messages
            )

            print(f"  Stop reason: {response.stop_reason}")

            if response.stop_reason == "end_turn":
                # Agente terminó
                final_text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        final_text += block.text

                print(f"\n[RESULTADO {agent_name}]")
                print(final_text[:500] + "..." if len(final_text) > 500 else final_text)
                return final_text

            elif response.stop_reason == "tool_use":
                # Agregar respuesta del asistente
                messages.append({"role": "assistant", "content": response.content})

                # Procesar tool calls
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        print(f"  Usando: {block.name}")

                        result = process_tool_call(block.name, block.input)

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result, ensure_ascii=False)
                        })

                messages.append({"role": "user", "content": tool_results})

            else:
                print(f"  Stop reason inesperado: {response.stop_reason}")
                break

        return "Max turns alcanzados sin respuesta final"

    def analyze_player_prop(self, player_name: str, opponent: str,
                            stat: str, line: float, odds: float,
                            is_home: bool = True) -> dict:
        """
        Análisis completo de una apuesta de Player Prop.

        Ejecuta los 3 agentes en secuencia:
        1. News Agent - Recopila noticias
        2. Lineup Agent - Verifica alineaciones
        3. Manager Agent - Toma decisión final
        """
        print("\n" + "="*80)
        print(f"ANÁLISIS: {player_name} {stat.upper()} O/U {line} @ {odds}")
        print("="*80)

        # === AGENTE 1: NEWS AGENT ===
        news_system = """Eres el News Agent, especialista en recopilar información
actualizada sobre jugadores de la NBA. Tu trabajo es:
1. Buscar noticias recientes sobre el jugador
2. Identificar lesiones, restricciones de minutos, o problemas de salud
3. Evaluar el nivel de riesgo (LOW, MEDIUM, HIGH)

Sé conciso y enfócate en información que afecte el rendimiento del jugador."""

        news_prompt = f"""Analiza las noticias recientes sobre {player_name}.
Busca información sobre:
- Estado de salud y lesiones
- Restricciones de minutos
- Descansos programados
- Cualquier factor que afecte su rendimiento

El jugador va a enfrentar a {opponent}."""

        news_result = self.run_agent(
            agent_name="NEWS_AGENT",
            system_prompt=news_system,
            user_prompt=news_prompt,
            tools=[SEARCH_NEWS_TOOL]
        )

        # === AGENTE 2: LINEUP AGENT ===
        lineup_system = """Eres el Lineup Agent, experto en analizar alineaciones
y cambios de roster en la NBA. Tu trabajo es:
1. Verificar qué jugadores están OUT, DOUBTFUL, QUESTIONABLE
2. Analizar cómo las bajas afectan el Usage Rate del jugador objetivo
3. Identificar si hay oportunidades o riesgos por cambios en el lineup

Sé específico sobre cómo los cambios de lineup afectan las estadísticas."""

        # Determinar el equipo del jugador (simplificado)
        team_abbrev = opponent  # En producción, obtener el equipo real del jugador

        lineup_prompt = f"""Analiza el lineup para el partido donde {player_name}
enfrentará a {opponent}.

Verifica:
1. El estado del equipo rival ({opponent})
2. Si hay bajas importantes que afecten el matchup
3. Cómo esto impacta las expectativas de {stat} para {player_name}"""

        lineup_result = self.run_agent(
            agent_name="LINEUP_AGENT",
            system_prompt=lineup_system,
            user_prompt=lineup_prompt,
            tools=[CHECK_LINEUP_TOOL, GET_PLAYER_STATS_TOOL]
        )

        # === AGENTE 3: MANAGER AGENT ===
        manager_system = """Eres el Manager Agent, el decisor final del sistema de apuestas.
Tu trabajo es:
1. Analizar la información del News Agent y Lineup Agent
2. Evaluar el Expected Value de la apuesta usando el modelo matemático
3. Tomar una decisión CONSERVADORA (solo apostar si el edge es claro)
4. Justificar tu decisión con datos concretos

IMPORTANTE: Solo recomienda apostar (OVER o UNDER) si:
- El Expected Value es positivo (>5%)
- No hay riesgos significativos (lesiones, restricciones)
- La información es consistente

Si hay CUALQUIER duda, recomienda NO_BET."""

        manager_prompt = f"""ANÁLISIS DE APUESTA
==================
Jugador: {player_name}
Estadística: {stat.upper()}
Línea: {line}
Cuota: {odds}
Local: {"Sí" if is_home else "No"}
Rival: {opponent}

INFORME DEL NEWS AGENT:
{news_result}

INFORME DEL LINEUP AGENT:
{lineup_result}

INSTRUCCIONES:
1. Usa la herramienta evaluate_betting_value para obtener la predicción del modelo
2. Combina esa información con los reportes de los otros agentes
3. Toma una decisión final usando make_final_decision

Sé conservador. Solo recomienda apostar si el edge es claro y los riesgos son bajos."""

        manager_result = self.run_agent(
            agent_name="MANAGER_AGENT",
            system_prompt=manager_system,
            user_prompt=manager_prompt,
            tools=[EVALUATE_BET_TOOL, GET_PLAYER_STATS_TOOL, MAKE_FINAL_DECISION_TOOL]
        )

        return {
            "player": player_name,
            "stat": stat,
            "line": line,
            "odds": odds,
            "news_analysis": news_result,
            "lineup_analysis": lineup_result,
            "final_decision": manager_result,
            "timestamp": datetime.now().isoformat()
        }


# =============================================================================
# CLI
# =============================================================================

def main():
    """Función principal para ejecutar el sistema de agentes."""
    import argparse

    parser = argparse.ArgumentParser(description="Sistema de Agentes NBA Props")
    parser.add_argument("--player", type=str, required=True, help="Nombre del jugador")
    parser.add_argument("--opponent", type=str, required=True, help="Rival (abreviatura)")
    parser.add_argument("--stat", type=str, default="pts", choices=["pts", "reb", "ast"])
    parser.add_argument("--line", type=float, required=True, help="Línea de apuesta")
    parser.add_argument("--odds", type=float, default=1.90, help="Cuota decimal")
    parser.add_argument("--home", action="store_true", help="Juega de local")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514",
                        help="Modelo Claude a usar")

    args = parser.parse_args()

    system = NBABettingAgentSystem(model=args.model)

    result = system.analyze_player_prop(
        player_name=args.player,
        opponent=args.opponent,
        stat=args.stat,
        line=args.line,
        odds=args.odds,
        is_home=args.home
    )

    print("\n" + "="*80)
    print("RESULTADO FINAL")
    print("="*80)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
