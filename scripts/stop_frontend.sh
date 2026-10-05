#!/bin/bash
# Ontology Builder Frontend 중지 (Next.js, port 9275)
#
# start_frontend.sh 는 있는데 stop 이 없었다 — 전체 종료가 콘솔만 남기고
# 끝나면 다음 기동이 "이미 응답 중"으로 건너뛰어 **낡은 빌드가 남는다**.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$(dirname "$SCRIPT_DIR")/logs/ontology_frontend.pid"
PORT=9275

STOPPED=false
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    kill "$(cat "$PID_FILE")" 2>/dev/null && STOPPED=true
fi
rm -f "$PID_FILE"

# `npm run dev` 는 자식(next-server)이 포트를 잡는다 — 부모만 죽이면 포트가
# 계속 응답해 "종료됐다"가 거짓이 된다. 포트 점유자를 마저 정리한다.
for _ in $(seq 1 5); do
    PID=$(lsof -ti :"$PORT" 2>/dev/null) || true
    [ -z "$PID" ] && break
    kill $PID 2>/dev/null && STOPPED=true
    sleep 1
done

if $STOPPED; then
    echo "Ontology Builder Frontend 중지됨"
else
    echo "실행 중인 프론트엔드 없음"
fi
