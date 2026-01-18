# Deployment Guide - Render

## Deploy to Render (Free Tier)

### Step 1: Prepare Your Repository

1. Make sure all changes are committed and pushed to GitHub:
```bash
git add .
git commit -m "Add Render deployment configuration"
git push
```

### Step 2: Create Render Account

1. Go to https://render.com
2. Sign up with your GitHub account (free)
3. Authorize Render to access your repositories

### Step 3: Deploy from Dashboard

#### Option A: Using render.yaml (Recommended)

1. Click **"New +"** → **"Blueprint"**
2. Connect your GitHub repository: `TheImageTokenCompany`
3. Render will automatically detect `render.yaml`
4. Click **"Apply"**

#### Option B: Manual Setup

1. Click **"New +"** → **"Web Service"**
2. Connect your GitHub repository
3. Configure:
   - **Name**: `image-token-company`
   - **Region**: Oregon (US West)
   - **Branch**: `main`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Plan**: Free

### Step 4: Set Environment Variables

In your Render service dashboard:

1. Go to **"Environment"** tab
2. Add these variables:
   - `GEMINI_API_KEY` = your_gemini_api_key
   - `OPENAI_API_KEY` = your_openai_api_key (optional)
   - `PYTHON_VERSION` = 3.11.0

3. Click **"Save Changes"**

### Step 5: Deploy

1. Render will automatically start building
2. First deploy takes ~10-15 minutes (downloading ML models)
3. Watch the build logs for any errors
4. Once deployed, you'll get a URL like: `https://image-token-company.onrender.com`

## Important Notes

### Free Tier Limitations

- **Spins down after 15 minutes of inactivity**
  - First request after sleep will be slow (30-60 seconds cold start)
  - Subsequent requests are fast

- **750 hours/month** (enough for one app running 24/7)

- **Limited Resources**
  - 512MB RAM (tight for ML models)
  - Shared CPU
  - May struggle with large images or multiple concurrent users

### Performance Tips

1. **Keep service alive**: Use a service like UptimeRobot to ping your app every 5 minutes
   - Sign up at https://uptimerobot.com (free)
   - Add monitor: `https://your-app.onrender.com`
   - Check interval: 5 minutes

2. **Optimize for memory**:
   - The app may hit RAM limits with all vision modules
   - Consider disabling some heavy modules if needed

3. **Monitor logs**:
   - Check Render dashboard for errors
   - Memory issues will show as crashes

## Troubleshooting

### Build Fails

**Error: "Out of memory during build"**
- ML dependencies are large
- Solution: Render free tier should handle it, but may take time
- If persistent, consider removing some vision modules

**Error: "Build timeout"**
- First build can take 15+ minutes
- Just wait, it will complete

### Runtime Errors

**Error: "Application failed to start"**
- Check environment variables are set
- Verify `GEMINI_API_KEY` is correct
- Check logs in Render dashboard

**Error: "Out of memory"**
- Free tier has 512MB RAM
- Disable heavy modules (YOLOv8, SmolVLM) if needed
- Or upgrade to paid tier ($7/month for 2GB RAM)

### Cold Starts

**App is slow on first request**
- Normal for free tier (spins down after 15 min)
- Use UptimeRobot to keep it alive
- Or accept the trade-off for free hosting

## Alternative: Local Development

If deployment is too slow or resource-constrained:

```bash
# Run locally
python app.py

# Access at http://localhost:5000
```

## Upgrading

If you need better performance:

- **Render Starter Plan**: $7/month
  - 2GB RAM
  - No sleep on inactivity
  - Faster CPU

- **Railway**: $5/month (pay-as-you-go)
  - More resources
  - Better for ML workloads

## Support

- Render Docs: https://render.com/docs
- Render Community: https://community.render.com
- GitHub Issues: https://github.com/JRNukala/TheImageTokenCompany/issues
