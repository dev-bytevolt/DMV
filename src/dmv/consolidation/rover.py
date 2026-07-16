from __future__ import annotations

from collections import defaultdict

GAP = "-"


def _nw_align(left: str, right: str) -> tuple[str, str]:
    m, n = len(left), len(right)
    if m == 0:
        return GAP * n, right
    if n == 0:
        return left, GAP * m

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            mismatch = 0 if left[i - 1] == right[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j - 1] + mismatch,
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
            )

    aligned_left: list[str] = []
    aligned_right: list[str] = []
    i, j = m, n
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and dp[i][j]
            == dp[i - 1][j - 1] + (0 if left[i - 1] == right[j - 1] else 1)
        ):
            aligned_left.append(left[i - 1])
            aligned_right.append(right[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            aligned_left.append(left[i - 1])
            aligned_right.append(GAP)
            i -= 1
        else:
            aligned_left.append(GAP)
            aligned_right.append(right[j - 1])
            j -= 1

    return "".join(reversed(aligned_left)), "".join(reversed(aligned_right))


def align_strings(strings: list[str]) -> list[str]:
    """Align strings with gap characters so they share the same width."""
    if not strings:
        return []
    if len(strings) == 1:
        return list(strings)

    # Progressive backbone alignment.
    # The previous implementation attempted to "remap" gaps but ended up
    # producing inconsistent character-to-column mappings, which can stitch
    # VIN/name-like strings into garbage.
    rows: list[str] = [strings[0]]
    backbone_aligned = strings[0]  # grows gaps as we proceed

    for text in strings[1:]:
        old_backbone_aligned = backbone_aligned
        backbone_plain = old_backbone_aligned.replace(GAP, "")

        aligned_backbone, aligned_text = _nw_align(backbone_plain, text)

        # Map backbone plain character index -> old column index.
        plain_idx_to_old_col: dict[int, int] = {}
        k = 0
        for col, ch in enumerate(old_backbone_aligned):
            if ch == GAP:
                continue
            plain_idx_to_old_col[k] = col
            k += 1

        # For each column in the new backbone alignment, record the backbone
        # plain character index (or None for gap columns).
        col_to_plain_idx: list[int | None] = []
        k = 0
        for ch in aligned_backbone:
            if ch == GAP:
                col_to_plain_idx.append(None)
            else:
                col_to_plain_idx.append(k)
                k += 1

        # Update previously aligned rows to the new backbone alignment.
        new_rows: list[str] = []
        for row in rows:
            chars: list[str] = []
            for col, plain_idx in enumerate(col_to_plain_idx):
                if plain_idx is None:
                    chars.append(GAP)
                else:
                    chars.append(row[plain_idx_to_old_col[plain_idx]])
            new_rows.append("".join(chars))

        rows = new_rows + [aligned_text]
        backbone_aligned = aligned_backbone

    return rows


def _normalize_vin(v: str) -> str:
    v = v.strip().upper()
    # Keep only alphanumerics; OCR often inserts spaces/punctuation.
    return "".join(ch for ch in v if ch.isalnum())


def vin_consensus(hypotheses: list[tuple[str, float]]) -> str:
    """
    VIN-specific consensus.

    Pick an actual source VIN rather than synthesizing a new string.
    Score each unique normalized VIN by total source confidence and how many
    documents reported the same value.
    """
    normalized: list[tuple[str, float]] = []
    for v, c in hypotheses:
        v_norm = _normalize_vin(v)
        if v_norm:
            normalized.append((v_norm, c))

    if not normalized:
        return ""
    if len(normalized) == 1:
        return normalized[0][0]

    vins_17 = [(v, c) for v, c in normalized if len(v) == 17]
    candidates = vins_17 or normalized

    total_confidence: dict[str, float] = defaultdict(float)
    occurrence_count: dict[str, int] = defaultdict(int)
    peak_confidence: dict[str, float] = defaultdict(float)
    for value, confidence in candidates:
        total_confidence[value] += confidence
        occurrence_count[value] += 1
        peak_confidence[value] = max(peak_confidence[value], confidence)

    return max(
        total_confidence.keys(),
        key=lambda value: (
            total_confidence[value],
            occurrence_count[value],
            peak_confidence[value],
        ),
    )


def rover_consensus(hypotheses: list[tuple[str, float]]) -> str:
    """Combine multiple OCR hypotheses with confidence-weighted voting."""
    def _normalize(v: str) -> str:
        # Make comparison robust to OCR casing and weird whitespace.
        v = v.strip()
        v = " ".join(v.split())
        return v.upper()

    filtered = [
        (_normalize(value), confidence)
        for value, confidence in hypotheses
        if value and value.strip()
    ]
    if not filtered:
        return ""
    if len(filtered) == 1:
        return filtered[0][0]

    # Guardrail: avoid stitching unrelated strings into Frankenstein output.
    # Use the highest-confidence hypothesis as a seed and keep only hypotheses
    # close to it by normalized edit-distance ratio.
    seed_value, _ = max(filtered, key=lambda item: item[1])

    def _levenshtein(a: str, b: str) -> int:
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j - 1] + cost,
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                )
        return dp[m][n]

    similarity_threshold = 0.75
    clustered: list[tuple[str, float]] = []
    for v, c in filtered:
        denom = max(len(seed_value), len(v), 1)
        dist_ratio = _levenshtein(seed_value, v) / denom
        if dist_ratio <= similarity_threshold:
            clustered.append((v, c))

    if not clustered:
        clustered = filtered
    if len(clustered) == 1:
        return clustered[0][0]

    strings = [value for value, _ in clustered]
    weights = [confidence for _, confidence in clustered]
    aligned = align_strings(strings)

    consensus: list[str] = []
    width = len(aligned[0])
    for column in range(width):
        votes: dict[str, float] = defaultdict(float)
        for row_index, row in enumerate(aligned):
            character = row[column]
            if character == GAP:
                continue
            votes[character] += weights[row_index]
        if votes:
            consensus.append(max(votes.items(), key=lambda item: item[1])[0])

    return "".join(consensus).strip()
