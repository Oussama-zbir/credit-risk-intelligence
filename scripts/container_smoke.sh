#!/usr/bin/env bash
# Smoke-test the scoring image against a model trained here, on synthetic loans.
#
#     docker build -t credit-risk-service .
#     pip install -e . pytest -c constraints.txt
#     scripts/container_smoke.sh credit-risk-service
#
# 1. Trains a logistic-regression artifact on the test suite's synthetic frame
#    with the local interpreter, which must resolve to the image's pins, or the
#    artifact's scikit-learn check refuses it (that is the point of the pins).
# 2. Serves it from the image with a read-only root filesystem and a read-only
#    mount, then checks /health reports its model id and /v1/score returns a PD.
# 3. Serves a copy whose model.pkl no longer matches its manifest, and requires
#    the container to exit non-zero, saying why, instead of starting.
set -euo pipefail

image=${1:?usage: $0 IMAGE}
python=${PYTHON:-python}
port=${PORT:-18000}
name=credit-risk-smoke-$$
work=$(mktemp -d)

cleanup() {
    docker rm -f "$name" "$name-tampered" >/dev/null 2>&1 || true
    rm -rf "$work"
}
trap cleanup EXIT

fail() {
    echo "FAIL: $*" >&2
    docker logs "$name" 2>&1 | tail -20 >&2 || true
    exit 1
}

echo "--- training a synthetic artifact"
"$python" - "$work/loans.parquet" <<'EOF'
import sys

sys.path.insert(0, "tests")
from conftest import make_processed

make_processed(6_000).to_parquet(sys.argv[1])
EOF
"$python" -m credit_risk.train "$work/loans.parquet" \
    --train 2012-01:2014-12 --test 2015-07:2016-12 --out "$work/report.md" \
    --artifact-dir "$work/artifacts" --artifact-model logistic-regression >/dev/null
artifact=$(find "$work/artifacts" -mindepth 1 -maxdepth 1 -type d)
model_id=$(basename "$artifact")

echo "--- serving $model_id"
docker run -d --name "$name" --read-only --tmpfs /tmp \
    -p "127.0.0.1:$port:8000" -v "$artifact:/artifact:ro" "$image" >/dev/null
for _ in $(seq 1 30); do
    health=$(curl -fsS "http://127.0.0.1:$port/health" 2>/dev/null) && break
    [ "$(docker inspect -f '{{.State.Running}}' "$name")" = true ] || fail "container exited"
    sleep 1
done
[ -n "${health:-}" ] || fail "/health did not answer within 30s"
echo "$health"
[ "$("$python" -c 'import json,sys; print(json.load(sys.stdin)["model_id"])' <<<"$health")" \
    = "$model_id" ] || fail "/health reports another model"

scored=$(curl -fsS "http://127.0.0.1:$port/v1/score" -H 'content-type: application/json' -d '{
  "applications": [{
    "loan_amnt": 15000, "term_months": 60, "purpose": "small_business",
    "application_type": "Individual", "annual_inc": 42000,
    "verification_status": "Not Verified", "home_ownership": "RENT", "fico": 662,
    "credit_history_months": 50, "delinq_2yrs": 1, "open_acc": 6, "total_acc": 12,
    "pub_rec": 0, "revol_bal": 9000, "dti": 31.5, "revol_util": 88,
    "inq_last_6mths": 3}]}') || fail "/v1/score failed"
"$python" - "$model_id" "$scored" <<'EOF' || fail "unexpected /v1/score response: $scored"
import json
import sys

body = json.loads(sys.argv[2])
(result,) = body["results"]
assert body["model_id"] == sys.argv[1], body["model_id"]
assert 0 < result["probability_of_default"] < 1, result
# "small_business" is not a synthetic training level: scored, and reported.
assert result["unseen_categories"] == {"purpose": ["small_business"]}, result
print(f"PD {result['probability_of_default']:.4f}, unseen {result['unseen_categories']}")
EOF

echo "--- serving a tampered copy"
cp -R "$artifact" "$work/tampered"
printf '\0' >>"$work/tampered/model.pkl"
docker run -d --name "$name-tampered" -v "$work/tampered:/artifact:ro" "$image" >/dev/null
for _ in $(seq 1 30); do
    [ "$(docker inspect -f '{{.State.Running}}' "$name-tampered")" = false ] && break
    sleep 1
done
status=$(docker inspect -f '{{.State.Running}} {{.State.ExitCode}}' "$name-tampered")
[ "${status%% *}" = false ] || fail "tampered artifact: still running after 30s"
[ "${status##* }" != 0 ] || fail "tampered artifact: exited 0"
docker logs "$name-tampered" 2>&1 | grep -q "does not match the manifest" \
    || fail "tampered artifact: exit reason not logged"
echo "refused to start (exit ${status##* }): model.pkl does not match the manifest"

echo "--- container smoke test passed"
