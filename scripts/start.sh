#!/bin/bash
# Ontology Builder Server (FastAPI, port 9274)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONTOLOGY_ROOT="$(dirname "$SCRIPT_DIR")"
LOGOS_ROOT="$(dirname "$ONTOLOGY_ROOT")"
LOG_DIR="$ONTOLOGY_ROOT/logs"
PID_FILE="$LOG_DIR/ontology_server.pid"
PORT=9274

mkdir -p "$LOG_DIR"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "이미 실행 중 (PID $(cat "$PID_FILE"), port $PORT)"
    exit 0
fi

source "$LOGOS_ROOT/.venv/bin/activate"
export PYTHONPATH="$LOGOS_ROOT:$LOGOS_ROOT/ontology"

# 저장 엔진 설정(.env, gitignore 됨) — 있으면 로드. 비밀번호는 repo 에 없다.
if [ -f "$ONTOLOGY_ROOT/.env" ]; then
    set -a; . "$ONTOLOGY_ROOT/.env"; set +a
    echo "환경 로드: $ONTOLOGY_ROOT/.env"
fi

# 문서화된 기동 방식(`ONTOLOGY_IMAGE_ASSETS=true ./start.sh`)을 **기본값**으로
# 끌어들인다 — 사람이 매번 접두사를 기억해야 하는 설정은 언젠가 빠진다.
# 명시 지정(호출자 env 또는 .env)이 있으면 그것이 이긴다.
export ONTOLOGY_IMAGE_ASSETS="${ONTOLOGY_IMAGE_ASSETS:-true}"

# 포트 해제 대기 — 직전 프로세스가 포트를 놓기 전에 bind 하면 "address already
# in use" 로 새 프로세스가 죽는다(재시작 레이스). 최대 ~5초 기다린다.
for i in $(seq 1 10); do
    lsof -ti :"$PORT" >/dev/null 2>&1 || break
    [ "$i" = "1" ] && echo "포트 $PORT 해제 대기…"
    sleep 0.5
done

nohup uvicorn ontology.server.main:app \
    --host 0.0.0.0 --port "$PORT" \
    >> "$LOG_DIR/ontology_server.log" 2>&1 &

echo $! > "$PID_FILE"
echo "Ontology Builder Server 시작: http://localhost:$PORT (PID $(cat "$PID_FILE"))"
echo "로그: $LOG_DIR/ontology_server.log"
