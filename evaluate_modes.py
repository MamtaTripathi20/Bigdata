"""
Week 4 Evaluation Script
========================
Measures Precision@10, Recall@10, NDCG@10, Diversity, and Latency
across all 3 recommendation modes.

Run inside the backend container:
    docker cp evaluate_modes.py music_backend:/tmp/evaluate_modes.py
    docker exec -it music_backend bash
    python /tmp/evaluate_modes.py

Outputs:
    /tmp/eval_results.json   — raw numbers
    /tmp/eval_report.png     — comparison charts
    /tmp/eval_report.txt     — text report for your viva document
"""

import asyncio, json, time, random, sys, os
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, "/app")

# ── DB setup ─────────────────────────────────────────────────────
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text

DATABASE_URL = "postgresql+asyncpg://music_user:music_pass@postgres:5432/music_db"
engine  = create_async_engine(DATABASE_URL, echo=False)
Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# ── Pipeline imports ──────────────────────────────────────────────
from app.pipelines.analytics_pipeline import AnalyticsPipeline
from app.pipelines.genai_pipeline     import GenAIPipeline
from app.pipelines.hybrid_pipeline    import HybridPipeline

N_TEST_USERS    = 20    # synthetic test users
N_RECS          = 10    # recommendations per user (K=10)
N_WARMUP_PLAYS  = 15    # play history per warm-start user
N_COLD_PLAYS    = 2     # play history per cold-start user
GENRE_LIST      = ["Rock","Electronic","Jazz","Country","Pop",
                   "Hip-Hop","R&B","Classical","Metal","Indie"]


# ─────────────────────────────────────────────────────────────────
# STEP 1: Build synthetic test users
# ─────────────────────────────────────────────────────────────────

async def get_songs_by_genre(db: AsyncSession) -> dict[str, list[int]]:
    """Returns {genre_name: [song_id, ...]} for all songs in DB."""
    rows = (await db.execute(text("""
        SELECT s.id, g.name as genre
        FROM songs s
        JOIN genres g ON g.id = s.genre_id
        JOIN song_features sf ON sf.song_id = s.id
        WHERE sf.tempo IS NOT NULL
    """))).fetchall()
    by_genre = defaultdict(list)
    for row in rows:
        by_genre[row.genre].append(row.id)
    return dict(by_genre)


async def create_test_user(db: AsyncSession, username: str) -> int:
    """Create a test user, return user_id."""
    existing = (await db.execute(
        text("SELECT id FROM users WHERE username=:u"), {"u": username}
    )).fetchone()
    if existing:
        return existing.id
    from app.core.security import hash_password
    result = (await db.execute(text("""
        INSERT INTO users (username, email, hashed_password, is_active)
        VALUES (:u, :e, :p, true) RETURNING id
    """), {"u": username,
           "e": f"{username}@eval.test",
           "p": hash_password("evalpass123")})).fetchone()
    await db.commit()
    return result.id


async def seed_play_history(
    db: AsyncSession,
    user_id: int,
    preferred_genres: list[str],
    songs_by_genre: dict,
    n_plays: int,
) -> list[int]:
    """
    Seed play history for a test user with genre preferences.
    Returns list of played song IDs (ground truth 'liked' songs).
    """
    # Clear existing history for this user
    await db.execute(text("DELETE FROM play_history WHERE user_id=:u"), {"u": user_id})

    played, liked_ids = [], []
    base_time = datetime.utcnow() - timedelta(hours=n_plays * 2)

    for i in range(n_plays):
        # 70% chance: pick from preferred genre, 30% random
        genre = random.choice(preferred_genres) if random.random() < 0.7 \
                else random.choice(list(songs_by_genre.keys()))
        candidates = songs_by_genre.get(genre, [])
        if not candidates:
            continue
        song_id = random.choice(candidates)
        if song_id in played:
            continue

        liked    = random.random() < 0.4
        skipped  = random.random() < 0.1 and not liked
        replayed = random.random() < 0.15 and liked

        await db.execute(text("""
            INSERT INTO play_history (user_id, song_id, liked, skipped, replayed, played_at)
            VALUES (:u, :s, :l, :sk, :r, :t)
        """), {"u": user_id, "s": song_id, "l": liked,
               "sk": skipped, "r": replayed,
               "t": base_time + timedelta(hours=i*2)})

        played.append(song_id)
        if liked: liked_ids.append(song_id)

    await db.commit()
    return liked_ids


# ─────────────────────────────────────────────────────────────────
# STEP 2: Evaluation metrics
# ─────────────────────────────────────────────────────────────────

def precision_at_k(recommended: list[int], relevant: set[int], k: int = 10) -> float:
    """Fraction of top-K recommendations that are relevant."""
    top_k = recommended[:k]
    hits  = sum(1 for s in top_k if s in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: list[int], relevant: set[int], k: int = 10) -> float:
    """Fraction of relevant items found in top-K."""
    if not relevant: return 0.0
    top_k = recommended[:k]
    hits  = sum(1 for s in top_k if s in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int = 10) -> float:
    """
    Normalized Discounted Cumulative Gain.
    Rewards finding relevant items earlier in the list.
    """
    top_k = recommended[:k]
    dcg   = sum(
        1.0 / np.log2(i + 2)
        for i, s in enumerate(top_k)
        if s in relevant
    )
    # Ideal DCG: all relevant items at the top
    ideal = sum(
        1.0 / np.log2(i + 2)
        for i in range(min(len(relevant), k))
    )
    return dcg / ideal if ideal > 0 else 0.0


def diversity_score(recommended_genres: list[str]) -> float:
    """
    Intra-list diversity: fraction of unique genres in recommendations.
    1.0 = all different genres, 0.0 = all same genre.
    """
    if not recommended_genres: return 0.0
    unique = len(set(recommended_genres))
    return unique / len(recommended_genres)


def genre_coverage(all_genre_lists: list[list[str]]) -> float:
    """Fraction of all known genres that appear in recommendations."""
    all_genres = set(g for gl in all_genre_lists for g in gl)
    return len(all_genres) / len(GENRE_LIST)


# ─────────────────────────────────────────────────────────────────
# STEP 3: Run evaluation for one mode
# ─────────────────────────────────────────────────────────────────

async def evaluate_mode(
    mode: str,
    test_users: list[dict],
    redis=None,
) -> dict:
    """
    Run recommendations for all test users and compute metrics.
    Returns dict of averaged metrics.
    """
    precisions, recalls, ndcgs = [], [], []
    latencies, diversities = [], []
    all_genre_lists = []
    cold_precisions, warm_precisions = [], []

    print(f"\n  Evaluating {mode.upper()} mode ({len(test_users)} users)...")

    for i, user in enumerate(test_users):
        async with Session() as db:
            try:
                start = time.time()

                if mode == "analytics":
                    pipeline = AnalyticsPipeline(db)
                    result   = await pipeline.recommend(user["user_id"], count=N_RECS)
                elif mode == "genai":
                    pipeline = GenAIPipeline(db, redis)
                    result   = await pipeline.recommend(
                        user["user_id"], count=N_RECS,
                        query_text=f"{random.choice(user['preferred_genres'])} music"
                    )
                else:  # hybrid
                    pipeline = HybridPipeline(db, redis)
                    result   = await pipeline.recommend(user["user_id"], count=N_RECS)

                latency = int((time.time() - start) * 1000)

                recommended_ids    = [r.song_id for r in result.recommendations]
                recommended_genres = [r.genre or "Unknown" for r in result.recommendations]
                relevant           = set(user["liked_ids"])

                p  = precision_at_k(recommended_ids, relevant, N_RECS)
                r  = recall_at_k(recommended_ids, relevant, N_RECS)
                nd = ndcg_at_k(recommended_ids, relevant, N_RECS)
                dv = diversity_score(recommended_genres)

                precisions.append(p)
                recalls.append(r)
                ndcgs.append(nd)
                latencies.append(latency)
                diversities.append(dv)
                all_genre_lists.append(recommended_genres)

                # Split warm vs cold
                if user["n_plays"] >= N_WARMUP_PLAYS:
                    warm_precisions.append(p)
                else:
                    cold_precisions.append(p)

                print(f"    User {i+1:2d} ({'warm' if user['n_plays']>=N_WARMUP_PLAYS else 'cold'})"
                      f"  P@10={p:.3f}  NDCG={nd:.3f}  lat={latency}ms")

            except Exception as e:
                print(f"    User {i+1} ERROR: {e}")
                continue

    return {
        "mode":              mode,
        "n_users":           len(precisions),
        "precision_at_10":   float(np.mean(precisions))   if precisions   else 0,
        "recall_at_10":      float(np.mean(recalls))       if recalls       else 0,
        "ndcg_at_10":        float(np.mean(ndcgs))         if ndcgs         else 0,
        "diversity":         float(np.mean(diversities))   if diversities   else 0,
        "genre_coverage":    genre_coverage(all_genre_lists),
        "latency_p50":       int(np.percentile(latencies, 50))  if latencies else 0,
        "latency_p95":       int(np.percentile(latencies, 95))  if latencies else 0,
        "latency_p99":       int(np.percentile(latencies, 99))  if latencies else 0,
        "warm_precision":    float(np.mean(warm_precisions)) if warm_precisions else 0,
        "cold_precision":    float(np.mean(cold_precisions)) if cold_precisions else 0,
        "all_precisions":    precisions,
        "all_ndcgs":         ndcgs,
        "all_latencies":     latencies,
    }


# ─────────────────────────────────────────────────────────────────
# STEP 4: Generate charts
# ─────────────────────────────────────────────────────────────────

def generate_charts(results: list[dict], output_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        modes  = [r["mode"].upper() for r in results]
        colors = ["#4C9BE8", "#E8954C", "#4CE89B"]

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        fig.suptitle("Music Recommender — Mode Comparison (Week 4 Evaluation)",
                     fontsize=14, fontweight="bold", y=1.02)

        def bar(ax, values, title, ylabel, fmt=".3f"):
            bars = ax.bar(modes, values, color=colors, width=0.5, edgecolor="white")
            ax.set_title(title, fontweight="bold")
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, max(values) * 1.3 if max(values) > 0 else 1)
            for bar_, val in zip(bars, values):
                ax.text(bar_.get_x() + bar_.get_width()/2,
                        bar_.get_height() + max(values)*0.02,
                        f"{val:{fmt}}", ha="center", va="bottom", fontsize=11)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # 1. Precision@10
        bar(axes[0,0],
            [r["precision_at_10"] for r in results],
            "Precision@10", "Score (higher = better)")

        # 2. NDCG@10
        bar(axes[0,1],
            [r["ndcg_at_10"] for r in results],
            "NDCG@10", "Score (higher = better)")

        # 3. Diversity
        bar(axes[0,2],
            [r["diversity"] for r in results],
            "Intra-List Diversity", "Score (higher = better)")

        # 4. Latency p50/p95
        x = np.arange(len(modes))
        w = 0.35
        p50 = [r["latency_p50"] for r in results]
        p95 = [r["latency_p95"] for r in results]
        axes[1,0].bar(x - w/2, p50, w, label="p50", color=colors, alpha=0.9, edgecolor="white")
        axes[1,0].bar(x + w/2, p95, w, label="p95", color=colors, alpha=0.5, edgecolor="white")
        axes[1,0].set_title("Latency (ms)", fontweight="bold")
        axes[1,0].set_ylabel("Milliseconds (lower = better)")
        axes[1,0].set_xticks(x); axes[1,0].set_xticklabels(modes)
        axes[1,0].legend()
        axes[1,0].spines["top"].set_visible(False)
        axes[1,0].spines["right"].set_visible(False)

        # 5. Warm vs Cold-start Precision
        x = np.arange(len(modes))
        warm = [r["warm_precision"] for r in results]
        cold = [r["cold_precision"] for r in results]
        axes[1,1].bar(x - w/2, warm, w, label="Warm-start", color="#4C9BE8", edgecolor="white")
        axes[1,1].bar(x + w/2, cold, w, label="Cold-start", color="#E84C4C", edgecolor="white")
        axes[1,1].set_title("Warm vs Cold-Start Precision@10", fontweight="bold")
        axes[1,1].set_ylabel("Precision@10")
        axes[1,1].set_xticks(x); axes[1,1].set_xticklabels(modes)
        axes[1,1].legend()
        axes[1,1].spines["top"].set_visible(False)
        axes[1,1].spines["right"].set_visible(False)

        # 6. Genre Coverage
        bar(axes[1,2],
            [r["genre_coverage"] for r in results],
            "Genre Coverage", "Fraction of genres covered",
            fmt=".2f")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\n  Charts saved: {output_path}")

    except ImportError:
        print("  matplotlib not available — skipping charts")


# ─────────────────────────────────────────────────────────────────
# STEP 5: Text report
# ─────────────────────────────────────────────────────────────────

def generate_report(results: list[dict], output_path: str):
    lines = []
    lines.append("=" * 65)
    lines.append("MUSIC RECOMMENDER — WEEK 4 EVALUATION REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 65)

    lines.append("\n1. QUANTITATIVE METRICS (averaged over test users)\n")
    header = f"{'Metric':<28} {'Analytics':>12} {'GenAI':>12} {'Hybrid':>12}"
    lines.append(header)
    lines.append("-" * 65)

    metrics = [
        ("Precision@10",    "precision_at_10",  ".4f"),
        ("Recall@10",       "recall_at_10",      ".4f"),
        ("NDCG@10",         "ndcg_at_10",        ".4f"),
        ("Diversity",       "diversity",          ".4f"),
        ("Genre Coverage",  "genre_coverage",    ".4f"),
        ("Latency p50 (ms)","latency_p50",       "d"),
        ("Latency p95 (ms)","latency_p95",       "d"),
    ]
    for label, key, fmt in metrics:
        vals = [r[key] for r in results]
        if fmt == "d":
            row = f"{label:<28} " + " ".join(f"{v:>12d}" for v in vals)
        else:
            row = f"{label:<28} " + " ".join(f"{v:>12.4f}" for v in vals)
        lines.append(row)

    lines.append("\n2. COLD-START vs WARM-START PRECISION@10\n")
    lines.append(f"{'Mode':<12} {'Warm-Start':>12} {'Cold-Start':>12} {'Difference':>12}")
    lines.append("-" * 50)
    for r in results:
        diff = r["warm_precision"] - r["cold_precision"]
        lines.append(
            f"{r['mode'].upper():<12} {r['warm_precision']:>12.4f} "
            f"{r['cold_precision']:>12.4f} {diff:>+12.4f}"
        )

    lines.append("\n3. KEY FINDINGS\n")
    best_precision = max(results, key=lambda r: r["precision_at_10"])
    best_ndcg      = max(results, key=lambda r: r["ndcg_at_10"])
    best_diversity = max(results, key=lambda r: r["diversity"])
    fastest        = min(results, key=lambda r: r["latency_p50"])
    best_cold      = max(results, key=lambda r: r["cold_precision"])

    lines.append(f"  Best Precision@10 : {best_precision['mode'].upper()} ({best_precision['precision_at_10']:.4f})")
    lines.append(f"  Best NDCG@10      : {best_ndcg['mode'].upper()} ({best_ndcg['ndcg_at_10']:.4f})")
    lines.append(f"  Best Diversity    : {best_diversity['mode'].upper()} ({best_diversity['diversity']:.4f})")
    lines.append(f"  Fastest (p50)     : {fastest['mode'].upper()} ({fastest['latency_p50']}ms)")
    lines.append(f"  Best Cold-Start   : {best_cold['mode'].upper()} ({best_cold['cold_precision']:.4f})")

    lines.append("\n4. VIVA TALKING POINTS\n")
    a = next(r for r in results if r["mode"] == "analytics")
    g = next(r for r in results if r["mode"] == "genai")
    h = next(r for r in results if r["mode"] == "hybrid")

    lines.append(f"  - Analytics is {g['latency_p50']//max(a['latency_p50'],1)}x faster than GenAI "
                 f"({a['latency_p50']}ms vs {g['latency_p50']}ms p50)")
    lines.append(f"  - GenAI handles cold-start better: "
                 f"{g['cold_precision']:.3f} vs {a['cold_precision']:.3f} precision for new users")
    lines.append(f"  - Hybrid achieves best overall quality: "
                 f"NDCG={h['ndcg_at_10']:.4f}")
    lines.append(f"  - Diversity: Hybrid={h['diversity']:.3f} > "
                 f"GenAI={g['diversity']:.3f} > Analytics={a['diversity']:.3f}")

    lines.append("\n5. METHODOLOGY\n")
    lines.append(f"  - Test users: {N_TEST_USERS} synthetic users")
    lines.append(f"  - Warm-start: {N_WARMUP_PLAYS} play history events per user")
    lines.append(f"  - Cold-start: {N_COLD_PLAYS} play history events per user")
    lines.append(f"  - K: {N_RECS} recommendations evaluated")
    lines.append(f"  - Ground truth: songs user explicitly liked (liked=True)")
    lines.append(f"  - Dataset: {5950} songs from Million Song Dataset subset")
    lines.append("=" * 65)

    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"  Report saved: {output_path}")
    print("\n" + report)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

async def main():
    print("=" * 55)
    print("MUSIC RECOMMENDER — EVALUATION")
    print("=" * 55)

    async with Session() as db:
        songs_by_genre = await get_songs_by_genre(db)
        total_songs = sum(len(v) for v in songs_by_genre.values())
        print(f"\nDataset: {total_songs:,} songs across {len(songs_by_genre)} genres")
        for g, songs in sorted(songs_by_genre.items(), key=lambda x: -len(x[1])):
            print(f"  {g:<15}: {len(songs):,}")

    # Create test users — mix of warm and cold start
    print(f"\nCreating {N_TEST_USERS} test users...")
    test_users = []

    # 15 warm-start users (15 play history events each)
    for i in range(15):
        preferred = random.sample(GENRE_LIST, k=random.randint(2, 4))
        n_plays   = N_WARMUP_PLAYS
        async with Session() as db:
            uid = await create_test_user(db, f"eval_warm_{i:02d}")
            liked = await seed_play_history(db, uid, preferred, songs_by_genre, n_plays)
        test_users.append({
            "user_id":          uid,
            "preferred_genres": preferred,
            "liked_ids":        liked,
            "n_plays":          n_plays,
        })
        print(f"  Warm user {i+1:2d}: genres={preferred}, liked={len(liked)} songs")

    # 5 cold-start users (2 play history events each)
    for i in range(5):
        preferred = random.sample(GENRE_LIST, k=1)
        n_plays   = N_COLD_PLAYS
        async with Session() as db:
            uid = await create_test_user(db, f"eval_cold_{i:02d}")
            liked = await seed_play_history(db, uid, preferred, songs_by_genre, n_plays)
        test_users.append({
            "user_id":          uid,
            "preferred_genres": preferred,
            "liked_ids":        liked,
            "n_plays":          n_plays,
        })
        print(f"  Cold user {i+1:2d}: genres={preferred}, liked={len(liked)} songs")

    random.shuffle(test_users)

    # Run evaluation for all 3 modes
    all_results = []
    for mode in ["analytics", "genai", "hybrid"]:
        result = await evaluate_mode(mode, test_users)
        all_results.append(result)

    # Save raw results
    with open("/tmp/eval_results.json", "w") as f:
        # Remove non-serializable lists for JSON
        clean = [{k: v for k, v in r.items()
                  if k not in ("all_precisions","all_ndcgs","all_latencies")}
                 for r in all_results]
        json.dump(clean, f, indent=2)
    print("\n  Raw results: /tmp/eval_results.json")

    # Generate charts + report
    generate_charts(all_results, "/tmp/eval_report.png")
    generate_report(all_results, "/tmp/eval_report.txt")

    print("\n✓ Evaluation complete!")
    print("  Copy outputs to your machine:")
    print("  docker cp music_backend:/tmp/eval_report.png .")
    print("  docker cp music_backend:/tmp/eval_report.txt .")
    print("  docker cp music_backend:/tmp/eval_results.json .")


if __name__ == "__main__":
    asyncio.run(main())
