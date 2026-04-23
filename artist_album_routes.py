"""
Enhanced Artist + Album routes
===============================
Rich stats computed from real DB data:
  - Artist: total plays, unique listeners, like/skip rates,
            top songs, genre breakdown, audio fingerprint,
            achievements, discography
  - Album:  full tracklist, mood analysis, key distribution,
            tempo curve, audio stats
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.database import get_db
from app.core.security import get_current_user
from app.models.db_models import User

router = APIRouter()

KEY_NAMES  = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
MODE_NAMES = {0: "Minor", 1: "Major"}


# ── Artist Full Profile ───────────────────────────────────────────

@router.get("/artists/{artist_id}/full")
async def get_artist_full(
    artist_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Basic artist info
    artist = (await db.execute(text(
        "SELECT id, name, mb_id, created_at FROM artists WHERE id=:aid"
    ), {"aid": artist_id})).fetchone()
    if not artist:
        raise HTTPException(status_code=404, detail="Artist not found")

    # All songs with features + play stats
    songs_raw = (await db.execute(text("""
        SELECT
            s.id, s.title, s.play_count, s.duration_s, s.mb_id,
            al.id AS album_id, al.title AS album, al.release_year,
            g.name AS genre,
            sf.tempo, sf.loudness, sf.energy, sf.danceability,
            sf.valence, sf.key, sf.mfcc_1,
            COUNT(ph.id)                                        AS total_plays,
            COUNT(ph.id) FILTER (WHERE ph.liked = true)        AS likes,
            COUNT(ph.id) FILTER (WHERE ph.skipped = true)      AS skips,
            COUNT(DISTINCT ph.user_id)                         AS unique_listeners
        FROM songs s
        LEFT JOIN albums al       ON al.id = s.album_id
        LEFT JOIN genres g        ON g.id  = s.genre_id
        LEFT JOIN song_features sf ON sf.song_id = s.id
        LEFT JOIN play_history ph  ON ph.song_id = s.id
        WHERE s.artist_id = :aid
        GROUP BY s.id, al.id, al.title, al.release_year, g.name,
                 sf.tempo, sf.loudness, sf.energy, sf.danceability,
                 sf.valence, sf.key, sf.mfcc_1
        ORDER BY s.play_count DESC
    """), {"aid": artist_id})).fetchall()

    if not songs_raw:
        return {
            "artist_id": artist.id, "name": artist.name,
            "mb_id": artist.mb_id, "song_count": 0,
            "total_plays": 0, "genres": [], "top_songs": [],
            "albums": [], "audio_fingerprint": {},
            "achievements": [], "genre_breakdown": [],
            "career": {}
        }

    # Aggregate stats
    total_plays      = sum(s.total_plays or 0 for s in songs_raw)
    total_likes      = sum(s.likes or 0 for s in songs_raw)
    total_skips      = sum(s.skips or 0 for s in songs_raw)
    unique_listeners = (await db.execute(text("""
        SELECT COUNT(DISTINCT ph.user_id) FROM play_history ph
        JOIN songs s ON s.id=ph.song_id WHERE s.artist_id=:aid
    """), {"aid": artist_id})).scalar() or 0

    like_rate  = round(total_likes / total_plays * 100, 1) if total_plays > 0 else 0
    skip_rate  = round(total_skips / total_plays * 100, 1) if total_plays > 0 else 0

    # Genre breakdown
    genre_counts: dict[str, int] = {}
    for s in songs_raw:
        g = s.genre or "Unknown"
        genre_counts[g] = genre_counts.get(g, 0) + 1
    genre_breakdown = sorted(
        [{"genre": g, "count": c, "pct": round(c / len(songs_raw) * 100)}
         for g, c in genre_counts.items()],
        key=lambda x: -x["count"]
    )

    # Audio fingerprint (avg of valid features)
    def avg(field):
        vals = [getattr(s, field) for s in songs_raw if getattr(s, field) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    audio_fp = {
        "tempo":        avg("tempo"),
        "energy":       avg("energy"),
        "danceability": avg("danceability"),
        "valence":      avg("valence"),
        "loudness":     avg("loudness"),
    }

    # Key distribution
    key_dist: dict[str, int] = {}
    for s in songs_raw:
        if s.key is not None and 0 <= s.key <= 11:
            kn = KEY_NAMES[s.key]
            key_dist[kn] = key_dist.get(kn, 0) + 1

    # Albums
    seen_albums: dict[int, dict] = {}
    for s in songs_raw:
        if s.album_id and s.album_id not in seen_albums:
            seen_albums[s.album_id] = {
                "album_id": s.album_id, "title": s.album,
                "release_year": s.release_year, "song_count": 0
            }
        if s.album_id:
            seen_albums[s.album_id]["song_count"] += 1
    albums = sorted(seen_albums.values(),
                    key=lambda a: a["release_year"] or 9999)

    # Career span
    years = [a["release_year"] for a in albums if a["release_year"]]
    career = {
        "debut_year":  min(years) if years else None,
        "latest_year": max(years) if years else None,
        "active_years": (max(years) - min(years) + 1) if len(years) > 1 else None,
        "album_count": len(albums),
    }

    # Achievements
    achievements = []
    top_song = songs_raw[0] if songs_raw else None
    if top_song and top_song.play_count > 0:
        achievements.append({
            "icon": "🔥",
            "title": "Most played",
            "desc":  f'"{top_song.title}" with {top_song.play_count:,} plays',
        })
    if unique_listeners >= 10:
        achievements.append({
            "icon": "👥",
            "title": "Wide reach",
            "desc":  f"Heard by {unique_listeners} listeners",
        })
    if like_rate >= 30:
        achievements.append({
            "icon": "❤️",
            "title": "Fan favourite",
            "desc":  f"{like_rate}% like rate across all plays",
        })
    if skip_rate <= 10 and total_plays > 10:
        achievements.append({
            "icon": "⏭️",
            "title": "Low skip rate",
            "desc":  f"Only {skip_rate}% of plays are skipped",
        })
    if len(albums) >= 3:
        achievements.append({
            "icon": "💿",
            "title": "Prolific artist",
            "desc":  f"{len(albums)} albums in catalogue",
        })
    if audio_fp.get("energy") and audio_fp["energy"] > 0.7:
        achievements.append({
            "icon": "⚡",
            "title": "High energy",
            "desc":  f"Average energy score {audio_fp['energy']}",
        })
    if audio_fp.get("danceability") and audio_fp["danceability"] > 0.7:
        achievements.append({
            "icon": "🕺",
            "title": "Dance floor ready",
            "desc":  f"Average danceability {audio_fp['danceability']}",
        })
    if career.get("active_years") and career["active_years"] >= 10:
        achievements.append({
            "icon": "🏆",
            "title": "Veteran artist",
            "desc":  f"{career['active_years']} years of recorded music",
        })

    # Top songs
    top_songs = [
        {
            "song_id":   s.id,
            "title":     s.title,
            "album":     s.album,
            "album_id":  s.album_id,
            "genre":     s.genre,
            "duration_s": s.duration_s,
            "play_count": s.play_count,
            "total_plays": s.total_plays or 0,
            "likes":     s.likes or 0,
            "skips":     s.skips or 0,
            "tempo":     round(s.tempo, 1) if s.tempo else None,
            "energy":    round(s.energy, 3) if s.energy else None,
            "key":       KEY_NAMES[s.key] if s.key is not None and 0 <= s.key <= 11 else None,
        }
        for s in songs_raw[:20]
    ]

    return {
        "artist_id":       artist.id,
        "name":            artist.name,
        "mb_id":           artist.mb_id,
        "song_count":      len(songs_raw),
        "total_plays":     total_plays,
        "unique_listeners": unique_listeners,
        "like_rate":       like_rate,
        "skip_rate":       skip_rate,
        "genres":          [g["genre"] for g in genre_breakdown[:3]],
        "genre_breakdown": genre_breakdown,
        "audio_fingerprint": audio_fp,
        "key_distribution":  key_dist,
        "achievements":    achievements,
        "top_songs":       top_songs,
        "albums":          albums,
        "career":          career,
    }


# ── Album Full Info ───────────────────────────────────────────────

@router.get("/albums/{album_id}/full")
async def get_album_full(
    album_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    album = (await db.execute(text("""
        SELECT al.id, al.title, al.release_year,
               a.name AS artist, a.id AS artist_id
        FROM albums al JOIN artists a ON a.id=al.artist_id
        WHERE al.id=:aid
    """), {"aid": album_id})).fetchone()
    if not album:
        raise HTTPException(status_code=404, detail="Album not found")

    songs = (await db.execute(text("""
        SELECT
            s.id, s.title, s.duration_s, s.play_count,
            g.name AS genre,
            sf.tempo, sf.loudness, sf.energy, sf.danceability,
            sf.valence, sf.key, sf.mode,
            COUNT(ph.id)                                   AS total_plays,
            COUNT(ph.id) FILTER (WHERE ph.liked=true)      AS likes,
            COUNT(ph.id) FILTER (WHERE ph.skipped=true)    AS skips
        FROM songs s
        LEFT JOIN genres g         ON g.id=s.genre_id
        LEFT JOIN song_features sf ON sf.song_id=s.id
        LEFT JOIN play_history ph  ON ph.song_id=s.id
        WHERE s.album_id=:aid
        GROUP BY s.id, g.name, sf.tempo, sf.loudness, sf.energy,
                 sf.danceability, sf.valence, sf.key, sf.mode
        ORDER BY s.id
    """), {"aid": album_id})).fetchall()

    if not songs:
        return {
            "album_id": album.id, "title": album.title,
            "artist": album.artist, "artist_id": album.artist_id,
            "release_year": album.release_year, "song_count": 0,
            "songs": [], "stats": {}, "mood": {}, "achievements": []
        }

    # Album stats
    total_dur   = sum(s.duration_s or 0 for s in songs)
    total_plays = sum(s.total_plays or 0 for s in songs)
    total_likes = sum(s.likes or 0 for s in songs)
    total_skips = sum(s.skips or 0 for s in songs)

    def avg(field):
        vals = [getattr(s, field) for s in songs if getattr(s, field) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    # Mood analysis based on energy + valence
    avg_energy  = avg("energy")  or 0
    avg_valence = avg("valence") or 0
    if avg_energy > 0.6 and avg_valence > 0.6:
        mood_label = "Euphoric"
        mood_color = "#E8A84C"
    elif avg_energy > 0.6 and avg_valence <= 0.6:
        mood_label = "Intense"
        mood_color = "#E84C4C"
    elif avg_energy <= 0.6 and avg_valence > 0.6:
        mood_label = "Chill"
        mood_color = "#4CE8A8"
    else:
        mood_label = "Melancholic"
        mood_color = "#4C9BE8"

    # Key distribution
    key_dist: dict[str, int] = {}
    for s in songs:
        if s.key is not None and 0 <= s.key <= 11:
            key_dist[KEY_NAMES[s.key]] = key_dist.get(KEY_NAMES[s.key], 0) + 1

    # Most popular key
    dominant_key = max(key_dist, key=key_dist.get) if key_dist else None

    # Mode split
    major_count = sum(1 for s in songs if s.mode == 1)
    minor_count = sum(1 for s in songs if s.mode == 0)

    # Genre breakdown
    genre_counts: dict[str, int] = {}
    for s in songs:
        g = s.genre or "Unknown"
        genre_counts[g] = genre_counts.get(g, 0) + 1

    # Tempo range
    tempos = [s.tempo for s in songs if s.tempo]
    tempo_range = {
        "min": round(min(tempos), 1) if tempos else None,
        "max": round(max(tempos), 1) if tempos else None,
        "avg": round(sum(tempos) / len(tempos), 1) if tempos else None,
    }

    # Achievements
    achievements = []
    most_played = max(songs, key=lambda s: s.total_plays or 0)
    if (most_played.total_plays or 0) > 0:
        achievements.append({
            "icon": "🔥", "title": "Top track",
            "desc": f'"{most_played.title}" is the most played',
        })
    if total_dur > 3600:
        achievements.append({
            "icon": "⏱️", "title": "Extended play",
            "desc": f"{total_dur // 3600}h {(total_dur % 3600) // 60}m total runtime",
        })
    if total_likes > 0 and total_plays > 0:
        like_rate = round(total_likes / total_plays * 100, 1)
        if like_rate >= 25:
            achievements.append({
                "icon": "❤️", "title": "Beloved album",
                "desc": f"{like_rate}% like rate",
            })
    if len(songs) >= 10:
        achievements.append({
            "icon": "💿", "title": "Full length",
            "desc": f"{len(songs)} tracks",
        })
    if avg_energy and avg_energy > 0.7:
        achievements.append({
            "icon": "⚡", "title": "High energy",
            "desc": f"Average energy {avg_energy:.2f}",
        })

    def fmt_dur(s):
        if not s: return "—"
        return f"{s // 60}:{str(s % 60).zfill(2)}"

    return {
        "album_id":    album.id,
        "title":       album.title,
        "artist":      album.artist,
        "artist_id":   album.artist_id,
        "release_year": album.release_year,
        "song_count":  len(songs),
        "total_duration_s": total_dur,
        "total_plays": total_plays,
        "stats": {
            "avg_tempo":        avg("tempo"),
            "avg_energy":       avg("energy"),
            "avg_danceability": avg("danceability"),
            "avg_valence":      avg("valence"),
            "avg_loudness":     avg("loudness"),
            "tempo_range":      tempo_range,
            "dominant_key":     dominant_key,
            "key_distribution": key_dist,
            "major_songs":      major_count,
            "minor_songs":      minor_count,
            "genre_breakdown":  genre_counts,
            "like_rate": round(total_likes / total_plays * 100, 1) if total_plays > 0 else 0,
            "skip_rate": round(total_skips / total_plays * 100, 1) if total_plays > 0 else 0,
        },
        "mood": {
            "label": mood_label,
            "color": mood_color,
            "energy":  avg_energy,
            "valence": avg_valence,
        },
        "achievements": achievements,
        "songs": [
            {
                "song_id":    s.id,
                "title":      s.title,
                "genre":      s.genre,
                "duration_s": s.duration_s,
                "play_count": s.play_count,
                "total_plays": s.total_plays or 0,
                "likes":      s.likes or 0,
                "tempo":      round(s.tempo, 1) if s.tempo else None,
                "energy":     round(s.energy, 3) if s.energy else None,
                "valence":    round(s.valence, 3) if s.valence else None,
                "key":        KEY_NAMES[s.key] if s.key is not None and 0 <= s.key <= 11 else None,
                "mode":       MODE_NAMES.get(s.mode) if s.mode is not None else None,
                "loudness":   round(s.loudness, 1) if s.loudness else None,
            }
            for s in songs
        ],
    }
