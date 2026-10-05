#!/bin/bash
# Ontology grounded 에이전트 ACP 인스턴스 (port 8915)
#
# 🎯 왜 이 파일이 생겼나 (2026-08-21)
#
#   이 인스턴스는 코드베이스에 기동 커맨드가 **없었다**. 유일한 기록이
#   .claude/settings.local.json 의 퍼미션 항목이라 사람이 매번 손으로 env 를
#   조립했고, 2026-08-21 재구동 때 ACP_AGENTS_JSON 을 빠뜨려 **에이전트 0개로
#   조용히 떴다** — 서버는 "시작 성공"을 보고했고 포트도 응답했다.
#   메커니즘 B: 같은 acp_server 코드를 다른 AGENTS_DIR 로 띄운다.
#
#   에이전트가 0개면 이 스크립트는 **실패로 끝난다**. 포트 응답만으로
#   판정하면 그 사고가 그대로 재발한다 (포트는 멀쩡히 열린다).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONTOLOGY_ROOT="$(dirname "$SCRIPT_DIR")"
LOGOS_ROOT="$(dirname "$ONTOLOGY_ROOT")"
ACP_DIR="$LOGOS_ROOT/acp_server"
LOG_DIR="$ONTOLOGY_ROOT/logs"
PID_FILE="$LOG_DIR/ontology_agents.pid"
LOG_FILE="$LOG_DIR/ontology_agents.log"
PORT=8915

mkdir -p "$LOG_DIR"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "이미 실행 중 (PID $(cat "$PID_FILE"), port $PORT)"
    exit 0
fi

# 포트 해제 대기 — 직전 프로세스가 포트를 놓기 전에 bind 하면 죽는다.
for i in $(seq 1 10); do
    lsof -ti :"$PORT" >/dev/null 2>&1 || break
    [ "$i" = "1" ] && echo "포트 $PORT 해제 대기…"
    sleep 0.5
done

source "$LOGOS_ROOT/.venv/bin/activate"

# 온톨로지 .env — 에이전트는 9274 를 HTTP 로만 쓰지만, ONTOLOGY_BASE 같은
# 접속 설정이 여기 있을 수 있다 (PG 자격증명은 이 프로세스에 불필요).
if [ -f "$ONTOLOGY_ROOT/.env" ]; then
    set -a; . "$ONTOLOGY_ROOT/.env"; set +a
fi

cd "$ACP_DIR"

# ⚠️ 두 env 는 **짝**이다. DIR 만 주면 row 는 8888 것을(121개) 읽고 모듈은
#    ontology/agents 에서 찾아 전부 실패한다 → 등록 0개로 조용히 뜬다.
PID=$(ACP_AGENTS_DIR="$ONTOLOGY_ROOT/agents" \
      ACP_AGENTS_JSON="$ONTOLOGY_ROOT/agents/agents.json" \
      "$LOGOS_ROOT/scripts/daemonize.sh" "$LOG_FILE" \
      python standalone_acp_server.py --port "$PORT")
echo "$PID" > "$PID_FILE"

# 포트 응답 대기
UP=false
for _ in $(seq 1 20); do
    if curl -s -o /dev/null --max-time 2 "http://localhost:$PORT/" 2>/dev/null; then
        UP=true; break
    fi
    sleep 1
done
if ! $UP; then
    echo "❌ 기동 실패 — 포트 $PORT 무응답. 로그: $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi

# 🎯 포트가 아니라 **적재된 에이전트 수**로 최종 판정한다.
COUNT=$(curl -s --max-time 5 -X POST "http://localhost:$PORT/jsonrpc" \
    -H 'Content-Type: application/json' \
    -d '{"jsonrpc":"2.0","id":1,"method":"list_agents","params":{}}' 2>/dev/null \
    | python3 -c "import json,sys
try: print(len(json.load(sys.stdin).get('result',{}).get('agents') or []))
except Exception: print(0)")

if [ "$COUNT" -lt 1 ]; then
    echo "❌ 포트는 응답하지만 **에이전트가 0개**다 — ACP_AGENTS_DIR/JSON 짝을 확인하라."
    echo "   로그: $LOG_FILE"
    exit 1
fi

echo "Ontology 에이전트 ACP 시작: http://localhost:$PORT (PID $PID, 에이전트 ${COUNT}개)"
echo "로그: $LOG_FILE"
