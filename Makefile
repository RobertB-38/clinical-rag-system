.PHONY: install test run ingest eval eval-check serve-stack k8s-deploy k8s-load

install:
	pip install -r requirements.txt

test:
	python -m pytest -q

run:
	uvicorn app.main:app --reload --port 8000

ingest:
	python -m app.ingest.run

eval:
	python -m eval.run_eval

eval-check:            ## fail on regression (used in CI)
	python -m eval.run_eval --check

serve-stack:           ## API + Prometheus + Grafana locally
	docker compose -f docker-compose.yml -f docker-compose.observability.yml up --build

# --- local Kubernetes (k3s / minikube) ---
k8s-deploy:
	kubectl apply -f k8s/configmap.yaml
	kubectl apply -f k8s/secret.example.yaml   # replace with a real secret first
	kubectl apply -f k8s/deployment.yaml
	kubectl apply -f k8s/service.yaml
	kubectl apply -f k8s/hpa.yaml

k8s-load:              ## port-forward, then drive load to trigger the HPA
	@echo "Run: kubectl port-forward svc/clinical-rag 8080:80 &"
	k6 run -e BASE_URL=http://localhost:8080 load/k6-load-test.js
