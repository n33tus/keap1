Deployment upload steps

1. Push repository to GitHub.
2. Go to Render website: https://render.com
3. Create account and connect GitHub.
4. Click New + and select Blueprint.
5. Select your repository.
6. Render reads render.yaml and deploys automatically.

If Blueprint is not used:
1. New + -> Web Service.
2. Select repository.
3. Environment: Docker.
4. Root Directory: app/deploy_bundle
5. Deploy.
