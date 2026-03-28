QSAR Docker deploy bundle

Contents:
- app_public.py
- requirements.txt
- Dockerfile
- .dockerignore
- models/
- scaling/
- descriptors.xml
- tcu_logo.png
- render.yaml

Local Docker test (optional):
1) docker build -t keap1-qsar-app .
2) docker run --rm -p 10000:10000 keap1-qsar-app
3) Open http://localhost:10000

Online deploy on Render (recommended):
1) Push this repo to GitHub.
2) In Render, click New + -> Blueprint.
3) Select your repository.
4) Render auto-detects app/deploy_bundle/render.yaml and creates the Docker web service.
5) Wait for build and deploy.

Manual Render setup (if not using Blueprint):
1) New + -> Web Service.
2) Connect the same repository.
3) Environment: Docker.
4) Root Directory: app/deploy_bundle
5) Instance type: Free (or paid if you need more memory/performance).
6) Add env var: PORT=10000
7) Click Deploy Web Service.

Post-deploy checks:
1) App opens with model selection in sidebar.
2) No "No models found in the 'models' directory" error.
3) A sample SMILES prediction runs successfully.
