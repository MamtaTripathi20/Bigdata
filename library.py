"""
Library Routes
==============
Endpoints for:
  - Liked songs (GET /library/liked)
  - Song details + audio features (GET /library/songs/{song_id})
  - Artist profile (GET /library/artists/{artist_id})
  - Album info (GET /library/albums/{album_id})
  - Playlists CRUD (GET/POST /library/playlists)
  - Add/remove song from playlist
  - Public playlist search
"""
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select
from pydantic import BaseModel
import structlog

from app.database import get_db
from app.core.security import get_current_user
from app.models.db_models import User, Playlist, PlaylistSong, Song

logger = structlog.get_logger()
router = APIRouter()


# ── Pydantic schemas ──────────────────────────────────────────────

class SongDetail(BaseModel):
    song_id:      int
    title:        str
    artist:       str
    artist_id:    int
    album:        Optional[str] = None
    album_id:     Optional[int] = None
    genre:        Optional[str] = None
    duration_s:   Optional[int] = None
    play_count:   int
    cover_url:    Optional[str] = None
    preview_url:  Optional[str] = None
    # Audio features
    tempo:        Optional[float] = None
    loudness:     Optional[float] = None
    key:          Optional[int]   = None
    mode:         Optional[int]   = None
    time_signature: Optional[int] = None
    energy:       Optional[float] = None
    danceability: Optional[float] = None
    mfcc_1:       Optional[float] = None
    mfcc_2:       Optional[float] = None
    mfcc_3:       Optional[float] = None
    # Derived
    key_name:     Optional[str]   = None
    mode_name:    Optional[str]   = None
    liked:        bool = False
    times_played: int = 0

class ArtistProfile(BaseModel):
    artist_id:    int
    name:         str
    mb_id:        Optional[str]  = None
    song_count:   int
    genres:       list[str]
    top_songs:    list[dict]
    albums:       list[dict]

class AlbumInfo(BaseModel):
    album_id:     int
    title:        str
    artist:       str
    artist_id:    int
    release_year: Optional[int] = None
    song_count:   int
    songs:        list[dict]

class PlaylistCreate(BaseModel):
    name:      str
    is_public: bool = False

class PlaylistOut(BaseModel):
    id:         int
    name:       str
    is_public:  bool
    owner:      str
    song_count: int
    created_at: datetime

class AddSongRequest(BaseModel):
    song_id: int

KEY_NAMES  = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
MODE_NAMES = {0: "Minor", 1: "Major"}


# ── Liked Songs ───────────────────────────────────────────────────

@router.get("/liked", summary="Get all songs the user has liked")
async def get_liked_songs(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    rows = (await db.execute(text("""
        SELECT DISTINCT ON (s.id)
            s.id, s.title, a.name AS artist, s.artist_id,
            al.title AS album, g.name AS genre,
            s.play_count, ph.played_at,
            sf.tempo, sf.energy
        FROM play_history ph
        JOIN songs s ON s.id = ph.song_id
        JOIN artists a ON a.id = s.artist_id
        LEFT JOIN albums al ON al.id = s.album_id
        LEFT JOIN genres g ON g.id = s.genre_id
        LEFT JOIN song_features sf ON sf.song_id = s.id
        WHERE ph.user_id = :uid AND ph.liked = true
        ORDER BY s.id, ph.played_at DESC
    """), {"uid": current_user.id})).fetchall()

    return {
        "count": len(rows),
        "songs": [
            {
                "song_id":    r.id,
                "title":      r.title,
                "artist":     r.artist,
                "artist_id":  r.artist_id,
                "album":      r.album,
                "genre":      r.genre,
                "play_count": r.play_count,
                "liked_at":   r.played_at,
                "tempo":      r.tempo,
                "energy":     r.energy,
            }
            for r in rows
        ],
    }


# ── Song Detail ───────────────────────────────────────────────────

@router.get("/songs/{song_id}", response_model=SongDetail, summary="Full song info + audio features")
async def get_song_detail(
    song_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    row = (await db.execute(text("""
        SELECT
            s.id, s.title, s.play_count, s.duration_s,
            a.name AS artist, a.id AS artist_id,
            al.title AS album, al.id AS album_id,
            g.name AS genre,
            sf.tempo, sf.loudness, sf.key, sf.mode, sf.time_signature,
            sf.energy, sf.danceability,
            sf.mfcc_1, sf.mfcc_2, sf.mfcc_3
        FROM songs s
        JOIN artists a ON a.id = s.artist_id
        LEFT JOIN albums al ON al.id = s.album_id
        LEFT JOIN genres g ON g.id = s.genre_id
        LEFT JOIN song_features sf ON sf.song_id = s.id
        WHERE s.id = :sid
    """), {"sid": song_id})).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Song not found")

    # Check if user liked this song
    liked_row = (await db.execute(text("""
        SELECT COUNT(*) FROM play_history
        WHERE user_id=:uid AND song_id=:sid AND liked=true
    """), {"uid": current_user.id, "sid": song_id})).scalar()

    times_played = (await db.execute(text("""
        SELECT COUNT(*) FROM play_history
        WHERE user_id=:uid AND song_id=:sid AND skipped=false
    """), {"uid": current_user.id, "sid": song_id})).scalar()

    key_name = KEY_NAMES[row.key] if row.key is not None and 0 <= row.key <= 11 else None

    return SongDetail(
        song_id=row.id, title=row.title, artist=row.artist,
        artist_id=row.artist_id, album=row.album, album_id=row.album_id,
        genre=row.genre, duration_s=row.duration_s, play_count=row.play_count,
        tempo=row.tempo, loudness=row.loudness, key=row.key, mode=row.mode,
        time_signature=row.time_signature, energy=row.energy,
        danceability=row.danceability,
        mfcc_1=row.mfcc_1, mfcc_2=row.mfcc_2, mfcc_3=row.mfcc_3,
        key_name=key_name,
        mode_name=MODE_NAMES.get(row.mode) if row.mode is not None else None,
        liked=bool(liked_row),
        times_played=int(times_played or 0),
    )


# ── Artist Profile ────────────────────────────────────────────────

@router.get("/artists/{artist_id}", response_model=ArtistProfile)
async def get_artist_profile(
    artist_id: int,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    artist = (await db.execute(
        text("SELECT id, name, mb_id FROM artists WHERE id=:aid"),
        {"aid": artist_id}
    )).fetchone()
    if not artist:
        raise HTTPException(status_code=404, detail="Artist not found")

    songs = (await db.execute(text("""
        SELECT s.id, s.title, g.name as genre, s.play_count,
               al.title as album, s.duration_s
        FROM songs s
        LEFT JOIN genres g ON g.id=s.genre_id
        LEFT JOIN albums al ON al.id=s.album_id
        WHERE s.artist_id=:aid
        ORDER BY s.play_count DESC
    """), {"aid": artist_id})).fetchall()

    albums = (await db.execute(text("""
        SELECT id, title, release_year,
               (SELECT COUNT(*) FROM songs WHERE album_id=albums.id) as song_count
        FROM albums WHERE artist_id=:aid ORDER BY release_year DESC NULLS LAST
    """), {"aid": artist_id})).fetchall()

    genres = list({s.genre for s in songs if s.genre})

    return ArtistProfile(
        artist_id=artist.id,
        name=artist.name,
        mb_id=artist.mb_id,
        song_count=len(songs),
        genres=genres,
        top_songs=[
            {"song_id": s.id, "title": s.title, "genre": s.genre,
             "album": s.album, "play_count": s.play_count, "duration_s": s.duration_s}
            for s in songs[:20]
        ],
        albums=[
            {"album_id": a.id, "title": a.title,
             "release_year": a.release_year, "song_count": a.song_count}
            for a in albums
        ],
    )


# ── Album Info ────────────────────────────────────────────────────

@router.get("/albums/{album_id}", response_model=AlbumInfo)
async def get_album_info(
    album_id: int,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    album = (await db.execute(text("""
        SELECT al.id, al.title, al.release_year,
               a.name as artist, a.id as artist_id
        FROM albums al JOIN artists a ON a.id=al.artist_id
        WHERE al.id=:aid
    """), {"aid": album_id})).fetchone()

    if not album:
        raise HTTPException(status_code=404, detail="Album not found")

    songs = (await db.execute(text("""
        SELECT s.id, s.title, s.duration_s, s.play_count,
               g.name as genre, sf.tempo
        FROM songs s
        LEFT JOIN genres g ON g.id=s.genre_id
        LEFT JOIN song_features sf ON sf.song_id=s.id
        WHERE s.album_id=:aid
        ORDER BY s.id
    """), {"aid": album_id})).fetchall()

    return AlbumInfo(
        album_id=album.id, title=album.title,
        artist=album.artist, artist_id=album.artist_id,
        release_year=album.release_year, song_count=len(songs),
        songs=[
            {"song_id": s.id, "title": s.title, "duration_s": s.duration_s,
             "genre": s.genre, "tempo": s.tempo, "play_count": s.play_count}
            for s in songs
        ],
    )


# ── Playlists ─────────────────────────────────────────────────────

@router.get("/playlists", summary="Get user's playlists")
async def get_playlists(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    rows = (await db.execute(text("""
        SELECT p.id, p.name, p.is_public, p.created_at,
               COUNT(ps.id) as song_count
        FROM playlists p
        LEFT JOIN playlist_songs ps ON ps.playlist_id = p.id
        WHERE p.user_id = :uid
        GROUP BY p.id ORDER BY p.created_at DESC
    """), {"uid": current_user.id})).fetchall()

    return {
        "playlists": [
            {"id": r.id, "name": r.name, "is_public": r.is_public,
             "song_count": r.song_count, "created_at": r.created_at}
            for r in rows
        ]
    }


@router.post("/playlists", summary="Create a playlist")
async def create_playlist(
    body: PlaylistCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not body.name.strip():
        raise HTTPException(status_code=400, detail="Playlist name cannot be empty")

    result = (await db.execute(text("""
        INSERT INTO playlists (user_id, name, is_public)
        VALUES (:uid, :name, :pub) RETURNING id, name, is_public, created_at
    """), {"uid": current_user.id, "name": body.name.strip(),
           "pub": body.is_public})).fetchone()
    await db.commit()

    return {"id": result.id, "name": result.name,
            "is_public": result.is_public, "created_at": result.created_at,
            "song_count": 0}


@router.get("/playlists/{playlist_id}", summary="Get playlist with songs")
async def get_playlist(
    playlist_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    playlist = (await db.execute(text("""
        SELECT p.id, p.name, p.is_public, p.user_id, p.created_at,
               u.username as owner
        FROM playlists p JOIN users u ON u.id=p.user_id
        WHERE p.id=:pid AND (p.user_id=:uid OR p.is_public=true)
    """), {"pid": playlist_id, "uid": current_user.id})).fetchone()

    if not playlist:
        raise HTTPException(status_code=404, detail="Playlist not found")

    songs = (await db.execute(text("""
        SELECT s.id, s.title, a.name as artist, s.artist_id,
               g.name as genre, ps.position, ps.added_at
        FROM playlist_songs ps
        JOIN songs s ON s.id=ps.song_id
        JOIN artists a ON a.id=s.artist_id
        LEFT JOIN genres g ON g.id=s.genre_id
        WHERE ps.playlist_id=:pid
        ORDER BY ps.position
    """), {"pid": playlist_id})).fetchall()

    return {
        "id": playlist.id, "name": playlist.name,
        "is_public": playlist.is_public, "owner": playlist.owner,
        "created_at": playlist.created_at,
        "songs": [
            {"song_id": s.id, "title": s.title, "artist": s.artist,
             "artist_id": s.artist_id, "genre": s.genre,
             "position": s.position, "added_at": s.added_at}
            for s in songs
        ]
    }


@router.post("/playlists/{playlist_id}/songs", summary="Add song to playlist")
async def add_song_to_playlist(
    playlist_id: int,
    body: AddSongRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    playlist = (await db.execute(text(
        "SELECT id FROM playlists WHERE id=:pid AND user_id=:uid"
    ), {"pid": playlist_id, "uid": current_user.id})).fetchone()

    if not playlist:
        raise HTTPException(status_code=404, detail="Playlist not found")

    song = await db.get(Song, body.song_id)
    if not song:
        raise HTTPException(status_code=404, detail="Song not found")

    # Get next position
    max_pos = (await db.execute(text(
        "SELECT COALESCE(MAX(position),0) FROM playlist_songs WHERE playlist_id=:pid"
    ), {"pid": playlist_id})).scalar()

    try:
        await db.execute(text("""
            INSERT INTO playlist_songs (playlist_id, song_id, position)
            VALUES (:pid, :sid, :pos)
            ON CONFLICT (playlist_id, song_id) DO NOTHING
        """), {"pid": playlist_id, "sid": body.song_id, "pos": max_pos + 1})
        await db.commit()
    except Exception:
        await db.rollback()

    return {"status": "added", "playlist_id": playlist_id, "song_id": body.song_id}


@router.delete("/playlists/{playlist_id}/songs/{song_id}", summary="Remove song from playlist")
async def remove_song_from_playlist(
    playlist_id: int,
    song_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    playlist = (await db.execute(text(
        "SELECT id FROM playlists WHERE id=:pid AND user_id=:uid"
    ), {"pid": playlist_id, "uid": current_user.id})).fetchone()
    if not playlist:
        raise HTTPException(status_code=404, detail="Playlist not found")

    await db.execute(text(
        "DELETE FROM playlist_songs WHERE playlist_id=:pid AND song_id=:sid"
    ), {"pid": playlist_id, "sid": song_id})
    await db.commit()
    return {"status": "removed"}


@router.delete("/playlists/{playlist_id}", summary="Delete playlist")
async def delete_playlist(
    playlist_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    playlist = (await db.execute(text(
        "SELECT id FROM playlists WHERE id=:pid AND user_id=:uid"
    ), {"pid": playlist_id, "uid": current_user.id})).fetchone()
    if not playlist:
        raise HTTPException(status_code=404, detail="Playlist not found")

    await db.execute(text("DELETE FROM playlist_songs WHERE playlist_id=:pid"), {"pid": playlist_id})
    await db.execute(text("DELETE FROM playlists WHERE id=:pid"), {"pid": playlist_id})
    await db.commit()
    return {"status": "deleted"}


# ── Public Playlist Search ────────────────────────────────────────

@router.get("/playlists/search/public", summary="Search public playlists")
async def search_public_playlists(
    q: str = Query("", description="Search term"),
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    rows = (await db.execute(text("""
        SELECT p.id, p.name, p.created_at,
               u.username as owner,
               COUNT(ps.id) as song_count
        FROM playlists p
        JOIN users u ON u.id=p.user_id
        LEFT JOIN playlist_songs ps ON ps.playlist_id=p.id
        WHERE p.is_public=true
          AND (:q = '' OR LOWER(p.name) LIKE :qlike OR LOWER(u.username) LIKE :qlike)
        GROUP BY p.id, u.username
        ORDER BY song_count DESC, p.created_at DESC
        LIMIT 20
    """), {"q": q, "qlike": f"%{q.lower()}%"})).fetchall()

    return {
        "results": [
            {"id": r.id, "name": r.name, "owner": r.owner,
             "song_count": r.song_count, "created_at": r.created_at}
            for r in rows
        ]
    }
