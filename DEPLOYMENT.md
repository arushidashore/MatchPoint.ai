# DigitalOcean Deployment Guide

This guide will help you deploy your MatchPoint.ai Flask application to DigitalOcean.

## Prerequisites

1. DigitalOcean account with an App Platform app created
2. Your code pushed to a Git repository (GitHub, GitLab, or Bitbucket)
3. All the configuration files from this setup

## Deployment Options

### Option 1: App Platform (Recommended)

DigitalOcean App Platform is the easiest way to deploy your Flask app.

#### Steps:

1. **Connect your repository:**
   - Go to your DigitalOcean App Platform dashboard
   - Click "Create App"
   - Connect your Git repository
   - Select the branch you want to deploy

2. **Configure the app:**
   - **App Type:** Web Service
   - **Source Directory:** `/` (root)
   - **Build Command:** `chmod +x build.sh && ./build.sh`
   - **Run Command:** `gunicorn --config gunicorn.conf.py app:app`

3. **Environment Variables:**
   Add these environment variables in your DigitalOcean app settings:
   ```
   SECRET_KEY=your-super-secret-key-change-this
   JWT_SECRET_KEY=your-jwt-secret-key-change-this
   DATABASE_URL=sqlite:///app.db
   PORT=8080
   WEB_CONCURRENCY=2
   TIMEOUT=120
   THREADS=2
   MAX_REQUESTS=1000
   ```

4. **Deploy:**
   - Click "Create Resources"
   - Wait for the deployment to complete

### Option 2: Droplet with Docker

If you prefer using a Droplet with Docker:

1. **Create a Droplet:**
   - Choose Ubuntu 20.04 or newer
   - At least 2GB RAM (4GB recommended for TensorFlow)
   - Enable Docker

2. **Connect to your Droplet:**
   ```bash
   ssh root@your-droplet-ip
   ```

3. **Clone your repository:**
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

4. **Build and run with Docker:**
   ```bash
   docker build -t matchpoint-app .
   docker run -d -p 80:8080 --name matchpoint-app \
     -e SECRET_KEY="your-secret-key" \
     -e JWT_SECRET_KEY="your-jwt-secret-key" \
     matchpoint-app
   ```

## Common Issues and Solutions

### Issue 1: Build Timeout
**Problem:** Build takes too long and times out
**Solution:** 
- Increase build timeout in App Platform settings
- Consider using a smaller TensorFlow model
- Optimize Docker build with multi-stage builds

### Issue 2: Memory Issues
**Problem:** App crashes due to insufficient memory
**Solution:**
- Upgrade to a higher tier plan
- Reduce `WEB_CONCURRENCY` to 1
- Optimize TensorFlow model loading

### Issue 3: File Upload Issues
**Problem:** Video upload fails
**Solution:**
- Check `MAX_CONTENT_LENGTH` setting
- Ensure proper file permissions
- Consider using cloud storage for large files

### Issue 4: Database Issues
**Problem:** Database not persisting
**Solution:**
- Use a managed PostgreSQL database
- Configure proper database URL
- Run migrations on deployment

## Environment Variables Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `SECRET_KEY` | Flask secret key | Required |
| `JWT_SECRET_KEY` | JWT secret key | Required |
| `DATABASE_URL` | Database connection string | `sqlite:///app.db` |
| `PORT` | Server port | `8080` |
| `WEB_CONCURRENCY` | Number of worker processes | `2` |
| `TIMEOUT` | Request timeout in seconds | `120` |
| `THREADS` | Threads per worker | `2` |
| `MAX_REQUESTS` | Max requests per worker | `1000` |

## Monitoring and Maintenance

1. **Check logs:** Use DigitalOcean's logging dashboard
2. **Monitor performance:** Watch CPU and memory usage
3. **Update dependencies:** Regularly update requirements.txt
4. **Backup database:** If using managed database, enable backups

## Security Considerations

1. **Change default secrets:** Always use strong, unique secret keys
2. **Enable HTTPS:** Use DigitalOcean's free SSL certificates
3. **Limit file uploads:** Configure appropriate file size limits
4. **Use environment variables:** Never hardcode secrets

## Troubleshooting

If deployment fails:

1. Check the build logs for specific errors
2. Verify all environment variables are set
3. Ensure your repository contains all necessary files
4. Check that the Python version matches runtime.txt
5. Verify that all dependencies can be installed

## Support

For DigitalOcean-specific issues, check:
- [DigitalOcean App Platform Documentation](https://docs.digitalocean.com/products/app-platform/)
- [DigitalOcean Community](https://www.digitalocean.com/community)

For application-specific issues, check your application logs and ensure all dependencies are properly configured.
