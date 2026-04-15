"""scripts/test_vlm_concurrency.py

VLM API 동시성 테스트 스크립트.

Test 1: 점진적 동시성 테스트 → safe_n 결정
Test 2: safe_n 기준으로 실제 평가 워크로드 시뮬레이션
Test 3: Semaphore(n) vs asyncio.gather() 속도·안정성 비교

Usage:
    python scripts/test_vlm_concurrency.py
"""

import asyncio
import base64
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.core import settings

# ── 샘플 이미지 (첫 번째 waypoint jpg 사용) ───────────────────────────────
_img_candidates = list((PROJECT_ROOT / "data" / "images").rglob("*.jpg"))
if not _img_candidates:
    raise FileNotFoundError("No .jpg images found under data/images/")
SAMPLE_IMAGE = _img_candidates[0]


def _make_payload(img_b64: str) -> dict:
    return {
        "model": settings.qwen_vlm_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": 'Score 1-10. Respond only: {"score":7}',
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    },
                ],
            }
        ],
        "max_tokens": 20,
        "temperature": 0.1,
    }


def _headers() -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.qwen_vlm_api_key}",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Test 1: 점진적 동시성 → safe_n 결정
# ══════════════════════════════════════════════════════════════════════════════

async def test_progressive_concurrency(
    levels: list[int] | None = None,
) -> int:
    """N=4, 8, 12, … 순으로 동시 호출 수를 올려가며 성공률·속도를 측정.

    Returns:
        safe_n: 성공률 100%를 유지하는 최대 동시 호출 수.
    """
    if levels is None:
        levels = [4, 8, 12, 16, 20, 24, 36, 48]

    print("\n=== Test 1: 점진적 동시성 테스트 ===")
    img_b64 = base64.b64encode(SAMPLE_IMAGE.read_bytes()).decode()
    safe_n = levels[0]

    for n in levels:
        print(f"\n  [N={n}] {n}개 동시 요청 발사…")

        async def call(call_id: int) -> dict:
            t0 = time.time()
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        f"{settings.qwen_vlm_api_url}/chat/completions",
                        headers=_headers(),
                        json=_make_payload(img_b64),
                        timeout=60.0,
                    )
                    resp.raise_for_status()
                return {"id": call_id, "ok": True, "elapsed": time.time() - t0}
            except Exception as e:
                return {
                    "id": call_id,
                    "ok": False,
                    "error": type(e).__name__,
                    "msg": str(e)[:80],
                    "elapsed": time.time() - t0,
                }

        t0 = time.time()
        results = await asyncio.gather(*[call(i) for i in range(n)])
        total = time.time() - t0

        ok = sum(1 for r in results if r["ok"])
        avg = sum(r["elapsed"] for r in results if r["ok"]) / (ok or 1)
        fail_types = [r.get("error") for r in results if not r["ok"]]

        print(f"  성공: {ok}/{n}  총 시간: {total:.1f}s  평균 응답: {avg:.1f}s")
        if fail_types:
            print(f"  실패 유형: {fail_types}")

        if ok == n:
            safe_n = n
        else:
            print(f"  → 실패 발생, safe_n={safe_n}으로 확정")
            break

        if n != levels[-1]:
            print("  (3초 쿨다운…)")
            await asyncio.sleep(3)

    print(f"\n  ✓ safe_n = {safe_n}")
    return safe_n


# ══════════════════════════════════════════════════════════════════════════════
# Test 2: safe_n 기준 실제 평가 워크로드 시뮬레이션
# ══════════════════════════════════════════════════════════════════════════════

async def test_real_workload(safe_n: int, total_calls: int = 20) -> None:
    """safe_n 동시성으로 total_calls개의 VLM 호출을 Semaphore 제어 하에 실행."""
    print(f"\n=== Test 2: 실제 워크로드 시뮬레이션 (safe_n={safe_n}, total={total_calls}) ===")
    img_b64 = base64.b64encode(SAMPLE_IMAGE.read_bytes()).decode()
    semaphore = asyncio.Semaphore(safe_n)

    async def call(call_id: int) -> dict:
        t0 = time.time()
        try:
            async with semaphore:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        f"{settings.qwen_vlm_api_url}/chat/completions",
                        headers=_headers(),
                        json=_make_payload(img_b64),
                        timeout=60.0,
                    )
                    resp.raise_for_status()
            return {"id": call_id, "ok": True, "elapsed": time.time() - t0}
        except Exception as e:
            return {
                "id": call_id,
                "ok": False,
                "error": type(e).__name__,
                "elapsed": time.time() - t0,
            }

    t0 = time.time()
    results = await asyncio.gather(*[call(i) for i in range(total_calls)])
    total = time.time() - t0

    ok = sum(1 for r in results if r["ok"])
    avg = sum(r["elapsed"] for r in results if r["ok"]) / (ok or 1)
    throughput = ok / total

    print(f"  성공: {ok}/{total_calls}  총 시간: {total:.1f}s")
    print(f"  평균 응답: {avg:.1f}s  처리량: {throughput:.2f} req/s")
    if ok < total_calls:
        fail_types = [r.get("error") for r in results if not r["ok"]]
        print(f"  실패 유형: {fail_types}")


# ══════════════════════════════════════════════════════════════════════════════
# Test 3: Semaphore vs gather 속도·안정성 비교
# ══════════════════════════════════════════════════════════════════════════════

async def test_semaphore_vs_gather(n: int = 12) -> None:
    """Compare Semaphore(n) vs gather(n) — speed and stability."""
    print(f"\n=== Test 3: Semaphore vs gather at N={n} ===")

    img_b64 = base64.b64encode(SAMPLE_IMAGE.read_bytes()).decode()

    async def call(client: httpx.AsyncClient, call_id: int, semaphore: asyncio.Semaphore | None = None) -> dict:
        payload = _make_payload(img_b64)
        t0 = time.time()
        try:
            if semaphore:
                async with semaphore:
                    resp = await client.post(
                        f"{settings.qwen_vlm_api_url}/chat/completions",
                        headers=_headers(),
                        json=payload,
                        timeout=60.0,
                    )
            else:
                resp = await client.post(
                    f"{settings.qwen_vlm_api_url}/chat/completions",
                    headers=_headers(),
                    json=payload,
                    timeout=60.0,
                )
            resp.raise_for_status()
            return {"id": call_id, "ok": True, "elapsed": time.time() - t0}
        except Exception as e:
            return {
                "id": call_id,
                "ok": False,
                "error": type(e).__name__,
                "elapsed": time.time() - t0,
            }

    # --- Method A: asyncio.gather() 제한 없음 ---
    print(f"\n[A] asyncio.gather() — 제한 없이 {n}개 동시 발사")
    async with httpx.AsyncClient() as client:
        t0 = time.time()
        results_a = await asyncio.gather(*[call(client, i) for i in range(n)])
        total_a = time.time() - t0

    ok_a = sum(1 for r in results_a if r["ok"])
    print(
        f"  Success: {ok_a}/{n}  Total: {total_a:.1f}s  "
        f"Avg: {sum(r['elapsed'] for r in results_a if r['ok']) / (ok_a or 1):.1f}s"
    )

    await asyncio.sleep(3)

    # --- Method B: Semaphore(n//2) ---
    sem_size = n // 2
    print(f"\n[B] Semaphore({sem_size}) — {sem_size}개씩 나눠서 처리")
    semaphore = asyncio.Semaphore(sem_size)
    async with httpx.AsyncClient() as client:
        t0 = time.time()
        results_b = await asyncio.gather(
            *[call(client, i, semaphore) for i in range(n)]
        )
        total_b = time.time() - t0

    ok_b = sum(1 for r in results_b if r["ok"])
    print(
        f"  Success: {ok_b}/{n}  Total: {total_b:.1f}s  "
        f"Avg: {sum(r['elapsed'] for r in results_b if r['ok']) / (ok_b or 1):.1f}s"
    )

    await asyncio.sleep(3)

    # --- Method C: Semaphore(n) = gather와 동일한 동시성 + 재시도 ---
    print(f"\n[C] Semaphore({n}) — gather와 동일한 동시성, 재시도 로직 추가")
    semaphore_full = asyncio.Semaphore(n)
    async with httpx.AsyncClient() as client:
        t0 = time.time()

        async def call_with_retry(call_id: int, max_retry: int = 2) -> dict:
            r: dict = {}
            for attempt in range(max_retry + 1):
                r = await call(client, call_id, semaphore_full)
                if r["ok"] or attempt == max_retry:
                    return r
                await asyncio.sleep(0.5 * (attempt + 1))
            return r

        results_c = await asyncio.gather(*[call_with_retry(i) for i in range(n)])
        total_c = time.time() - t0

    ok_c = sum(1 for r in results_c if r["ok"])
    print(
        f"  Success: {ok_c}/{n}  Total: {total_c:.1f}s  "
        f"Avg: {sum(r['elapsed'] for r in results_c if r['ok']) / (ok_c or 1):.1f}s"
    )

    best_speed = min(
        ("A", total_a), ("B", total_b), ("C", total_c), key=lambda x: x[1]
    )[0]
    stability_note = "C가 가장 균형적" if ok_c >= ok_a else "A도 안정적"

    print(f"""
=== 결론 ===
A (gather 무제한):   {total_a:.1f}s  성공률 {ok_a / n:.0%}
B (Semaphore {sem_size:>2}):    {total_b:.1f}s  성공률 {ok_b / n:.0%}
C (Semaphore {n:>2}+retry): {total_c:.1f}s  성공률 {ok_c / n:.0%}

속도만 보면: {best_speed}
안정성 고려: {stability_note}
""")


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

async def main() -> None:
    print(f"VLM endpoint : {settings.qwen_vlm_api_url}")
    print(f"Model        : {settings.qwen_vlm_model}")
    print(f"Sample image : {SAMPLE_IMAGE.name}")

    # Test 1: 점진적 동시성 → safe_n
    safe_n = await test_progressive_concurrency()

    # Test 2: safe_n 기준 실제 워크로드
    await asyncio.sleep(3)
    await test_real_workload(safe_n)

    # Test 3: Semaphore vs gather
    test_n = min(safe_n, 48) if safe_n else 48
    await asyncio.sleep(3)
    await test_semaphore_vs_gather(test_n)


if __name__ == "__main__":
    asyncio.run(main())
