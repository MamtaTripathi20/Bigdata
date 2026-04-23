#!/bin/bash
# deploy_features.sh
# Run from inside the music_backend container after copying files with docker cp
# Usage: bash /tmp/deploy_features.sh

set -e

BACKEND_APP="/app/app"
FRONTEND_SRC="/app/frontend/src"   # not used — frontend files go via docker cp to host

echo "=== Deploying backend library routes ==="

# Add is_public column to playlists if not exists
python -c "
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

async def migrate():
    engine = create_async_engine('postgresql+asyncpg://music_user:music_pass@postgres:5432/music_db')
    async with engine.begin() as conn:
        # Add is_public column
        try:
            await conn.execute(text('ALTER TABLE playlists ADD COLUMN IF NOT EXISTS is_public BOOLEAN DEFAULT false'))
            print('✓ is_public column added to playlists')
        except Exception as e:
            print(f'  (already exists or error: {e})')
    await engine.dispose()

asyncio.run(migrate())
"

echo "✓ Migration done"

# Copy library.py to routes
cp /tmp/library.py $BACKEND_APP/api/routes/library.py
echo "✓ library.py copied"

# Patch main.py to include library router
python -c "
import re

with open('$BACKEND_APP/main.py', 'r') as f:
    content = f.read()

# Add import if not already there
if 'from app.api.routes import library' not in content:
    content = content.replace(
        'from app.api.routes import recommendations',
        'from app.api.routes import recommendations\nfrom app.api.routes import library as library_route'
    )

# Add router if not already there
if 'library_route.router' not in content:
    content = content.replace(
        'app.include_router(health.router,',
        'app.include_router(library_route.router, prefix=\"/api/v1/library\", tags=[\"Library\"])\napp.include_router(health.router,'
    )

with open('$BACKEND_APP/main.py', 'w') as f:
    f.write(content)

print('✓ main.py patched')
"

# Clear pyc cache
find $BACKEND_APP -name "*.pyc" -delete
find $BACKEND_APP -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
echo "✓ Cache cleared"

echo ""
echo "=== Backend deploy complete ==="
echo "Now restart: docker compose restart backend"
echo ""
echo "=== Frontend files to copy (run on your Windows machine) ==="
echo "docker cp SongCard.tsx music_frontend:/app/src/components/modes/SongCard.tsx"
echo "docker cp SongInfoModal.tsx music_frontend:/app/src/components/ui/SongInfoModal.tsx"
echo "docker cp PlaylistModal.tsx music_frontend:/app/src/components/ui/PlaylistModal.tsx"
echo "docker cp library_page.tsx music_frontend:/app/src/app/library/page.tsx"
echo "docker cp page.tsx music_frontend:/app/src/app/page.tsx"
echo "docker cp api.ts music_frontend:/app/src/lib/api.ts"
