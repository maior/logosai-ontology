#!/bin/bash
# Ontology grounded 에이전트 ACP 중지 (port 8915)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$(dirname "$SCRIPT_DIR")/logs/ontology_agents.pid"
PORT=8915

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    kill "$(cat "$PID_FILE")"
    rm -f "$PID_FILE"
    echo "Ontology 에이전트 ACP 중지됨"
else
    # PID 파일이 없거나 낡았으면 포트 기준으로 정리. 8888(메인 ACP)은 다른
    # 포트라 건드리지 않는다 — 같은 코드를 도는 두 인스턴스를 포트로 가른다.
    PID=$(lsof -ti :"$PORT" 2>/dev/null)
    if [ -n "$PID" ]; then
        kill $PID
        echo "Ontology 에이전트 ACP 중지됨 (port 기준, PID $PID)"
    else
        echo "실행 중인 에이전트 ACP 없음"
    fi
    rm -f "$PID_FILE"
fi
