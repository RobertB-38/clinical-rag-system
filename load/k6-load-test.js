// k6 load test for the Clinical RAG service.
//
//   k6 run -e BASE_URL=http://localhost:8080 load/k6-load-test.js
//
// Ramps virtual users to push CPU past the HPA's 60% target so you can watch
// `kubectl get hpa -w` add pods. Thresholds encode the SLOs in docs/SLOs.md;
// the run fails (non-zero exit) if they are breached, so it doubles as a gate.
import http from "k6/http";
import { check, sleep } from "k6";
import { Trend } from "k6/metrics";

const BASE_URL = __ENV.BASE_URL || "http://localhost:8080";
const answerLatency = new Trend("answer_latency_ms", true);

const QUESTIONS = [
  "What is the recommended timeframe for giving antibiotics in adult sepsis?",
  "How is a COPD diagnosis confirmed?",
  "How do you assess the severity of an acute asthma attack in adults?",
  "How is diabetic ketoacidosis managed in adults with type 1 diabetes?",
  "What are the core standard precautions for infection prevention and control?",
];

export const options = {
  scenarios: {
    ramp: {
      executor: "ramping-vus",
      startVUs: 1,
      stages: [
        { duration: "30s", target: 10 },
        { duration: "1m", target: 30 },
        { duration: "1m", target: 30 },
        { duration: "30s", target: 0 },
      ],
    },
  },
  thresholds: {
    http_req_failed: ["rate<0.01"],          // <1% errors
    http_req_duration: ["p(95)<8000"],       // p95 < 8s end-to-end
  },
};

export default function () {
  const q = QUESTIONS[Math.floor(Math.random() * QUESTIONS.length)];
  const res = http.post(
    `${BASE_URL}/v1/query`,
    JSON.stringify({ question: q, top_k: 8 }),
    { headers: { "Content-Type": "application/json" } }
  );
  answerLatency.add(res.timings.duration);
  check(res, {
    "status 200": (r) => r.status === 200,
    "has answer": (r) => r.json("answer") !== undefined,
  });
  sleep(1);
}
