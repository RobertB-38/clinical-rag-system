# Deploying to local Kubernetes (k3s / minikube)

This proves the Kubernetes skill — manifests, probes, an HPA, and autoscaling
under load. The public demo URL stays on the Hugging Face Space; this is the
"runs on a real orchestrator" proof, not a production cluster.

## 1. Build the image and load it into the cluster
```bash
docker build -t clinical-rag:latest .

# minikube:
minikube image load clinical-rag:latest
# k3s:
docker save clinical-rag:latest | sudo k3s ctr images import -
```

## 2. Create the secret (never commit a real one)
```bash
kubectl create secret generic clinical-rag-secrets \
  --from-literal=RAG_ANTHROPIC_API_KEY=sk-ant-... \
  --from-literal=OPENAI_API_KEY=sk-...
```

## 3. Apply manifests
```bash
make k8s-deploy        # configmap, deployment, service, hpa
kubectl rollout status deploy/clinical-rag
```

## 4. Drive load and watch autoscaling
```bash
kubectl port-forward svc/clinical-rag 8080:80 &
kubectl get hpa -w &           # watch replicas climb past 60% CPU
k6 run -e BASE_URL=http://localhost:8080 load/k6-load-test.js
```
You should see the HPA add pods during the 30-VU plateau and scale back down
after. Capture `kubectl get hpa` before/after for the README.

## Notes
- `metrics-server` must be installed for the HPA to read CPU
  (`minikube addons enable metrics-server`, or the k3s bundled one).
- The in-process rate limiter is per-pod; with multiple replicas a client can
  get up to `replicas × limit`. A shared Redis limiter is the production fix.
