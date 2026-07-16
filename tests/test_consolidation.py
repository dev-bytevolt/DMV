import json
from pathlib import Path

import pytest

from dmv.consolidation.field import (
    consolidate_field,
    compute_field_confidence,
    collect_variants,
    collect_source_variants,
    needs_manual_review,
)
from dmv.consolidation.rover import align_strings, rover_consensus, vin_consensus
from dmv.consolidation.service import (
    CONSOLIDATED_DATA_FILENAME,
    consolidate_extractions,
)


def test_rover_consensus_returns_single_hypothesis() -> None:
    assert rover_consensus([("  ABC  ", 0.8)]) == "ABC"


def test_rover_consensus_picks_agreement() -> None:
    hypotheses = [
        ("5NMMBDTL6TH064502", 0.98),
        ("5NMMBDTL6TH064502", 0.75),
        ("5NMMBDT6TH064502", 0.6),
    ]
    assert rover_consensus(hypotheses) == "5NMMBDTL6TH064502"


def test_rover_consensus_uses_confidence_weights() -> None:
    hypotheses = [
        ("AAA", 0.1),
        ("BBB", 0.95),
    ]
    assert rover_consensus(hypotheses) == "BBB"


def test_align_strings_inserts_gaps() -> None:
    aligned = align_strings(["ABC", "AC"])
    assert aligned == ["ABC", "A-C"]

def test_vin_consensus_filters_to_17_chars_and_votes_per_position() -> None:
    hypotheses = [
        ("5NMMADTB6TH0164502", 0.93),  # 18 chars; should be ignored
        ("5NMMBDTB6TH064502", 0.99),
        ("5NMMBDTB6LH064502", 0.83),  # plausible OCR variant
        ("5NMMB1DTB6TH064502", 0.98),
        ("KMTWBDBT6TH064502", 0.98),
        ("5NMMBDTB6TH064502", 0.95),
    ]
    assert vin_consensus(hypotheses) == "5NMMBDTB6TH064502"


def test_vin_consensus_prefers_repeated_source_value() -> None:
    hypotheses = [
        ("5NMMBDTB6TH064502", 0.99),
        ("5NMMBDTB6TH064502", 0.97),
        ("5NMMBDTB6TH064502", 0.84),
        ("5NMMBFDTB6TH064502", 0.98),
        ("5NMMBDBT6TH064502", 0.98),
    ]
    assert vin_consensus(hypotheses) == "5NMMBDTB6TH064502"


def test_vin_consensus_never_returns_unseen_value() -> None:
    hypotheses = [
        ("5NMMBDTB6TH064502", 0.99),
        ("5NMMBFDTB6TH064502", 0.98),
        ("5NMMBDBT6TH064502", 0.98),
    ]
    result = vin_consensus(hypotheses)
    assert result in {value for value, _ in hypotheses}


def test_rover_consensus_does_not_stitch_unrelated_strings() -> None:
    hypotheses = [
        ("Toms River", 0.98),
        ("COLLEGE ST", 0.78),
        ("COLLEGE STA", 0.92),
    ]
    assert rover_consensus(hypotheses) == "TOMS RIVER"


def test_collect_variants_deduplicates_by_normalized_value() -> None:
    variants = collect_variants(
        [
            ("GENESIS", 0.99),
            ("genesis", 0.5),
            ("GENES", 0.4),
        ]
    )
    assert len(variants) == 2
    assert variants[0].value == "GENESIS"
    assert variants[0].confidence == 0.99


def test_collect_source_variants_keeps_document_metadata() -> None:
    variants = collect_source_variants(
        [
            ("GENESIS", 0.99, "Retail Certificate of Sale Receipt", "retail_certificate_of_sale"),
            ("GENESIS", 0.95, "Dealer Invoice", "dealer_invoice"),
        ]
    )
    assert len(variants) == 2
    assert variants[0].source_document_name == "Retail Certificate of Sale Receipt"
    assert variants[0].source_document_type == "retail_certificate_of_sale"


def test_compute_field_confidence_penalizes_disagreement() -> None:
    unanimous = collect_variants([("GENESIS", 0.99), ("GENESIS", 0.95)])
    conflicting = collect_variants([("GENESIS", 0.99), ("HYUNDAI", 0.98)])

    unanimous_conf = compute_field_confidence("GENESIS", unanimous)
    conflicting_conf = compute_field_confidence("GENESIS", conflicting)

    assert unanimous_conf > conflicting_conf


def test_consolidate_field_includes_metadata() -> None:
    result = consolidate_field(
        [
            ("5NMMBDTL6TH064502", 0.98, "Certificate of Origin", "manufacturer_certificate"),
            ("5NMMBDTL6TH064502", 0.75, "Dealer Invoice", "dealer_invoice"),
            ("5NMMBDT6TH064502", 0.6, "Insurance Card", "insurance_card"),
        ],
        use_vin=True,
    )
    assert result is not None
    assert result.value == "5NMMBDTL6TH064502"
    assert len(result.variants) == 3
    assert result.variants[0].source_document_name == "Certificate of Origin"
    assert result.confidence > 0.7
    assert result.review_required is False


def test_consolidate_field_flags_review_for_conflicting_values() -> None:
    result = consolidate_field(
        [
            ("Toms River", 0.98, "GEICO Insurance", "insurance_card"),
            ("COLLEGE ST", 0.78, "Dealer Invoice", "dealer_invoice"),
            ("COLLEGE STA", 0.92, "Retail Certificate", "retail_certificate_of_sale"),
        ]
    )
    assert result is not None
    assert result.value == "TOMS RIVER"
    assert len(result.variants) == 3
    assert result.review_required is True


def test_needs_manual_review_when_consensus_does_not_match_any_variant() -> None:
    variants = collect_variants(
        [
            ("NAFTALI T SHAPIRO", 0.7),
            ("NAFTALI T. SHAPIRO", 0.8),
        ]
    )
    assert needs_manual_review("NAFTALI TZVI SHAPIRO", variants, 0.9) is True


def test_consolidate_extractions_merges_canonical_fields(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "bundle"
    extracted_dir = artifact_dir / "extracted"
    extracted_dir.mkdir(parents=True)

    (extracted_dir / "Certificate_of_Origin.json").write_text(
        json.dumps(
            {
                "document_type": "manufacturer_certificate",
                "source_document_name": "Certificate of Origin for a Vehicle",
                "vehicle_vin": {"value": "5NMMBDTL6TH064502", "confidence": 0.98},
                "vehicle_make": {"value": "GENESIS", "confidence": 0.99},
                "extra": [
                    {
                        "field_name": "plant_code",
                        "value": "A1",
                        "confidence": 0.8,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (extracted_dir / "Dealer_Invoice.json").write_text(
        json.dumps(
            {
                "document_type": "dealer_invoice",
                "source_document_name": "Dealer Invoice",
                "vehicle_vin": {"value": "5NMMBDTL6TH064502", "confidence": 0.95},
                "vehicle_make": {"value": "GENES", "confidence": 0.4},
                "extra": [
                    {
                        "field_name": "stock_number",
                        "value": "123",
                        "confidence": 0.9,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = consolidate_extractions(extracted_dir, artifact_dir)
    payload = json.loads(result.output_json.read_text(encoding="utf-8"))

    assert result.output_json.name == CONSOLIDATED_DATA_FILENAME
    assert result.field_count == 2
    assert payload["vehicle_vin"]["value"] == "5NMMBDTL6TH064502"
    assert len(payload["vehicle_vin"]["variants"]) == 2
    assert payload["vehicle_vin"]["variants"][0]["source_document_name"] == "Certificate of Origin for a Vehicle"
    assert payload["vehicle_vin"]["variants"][1]["source_document_name"] == "Dealer Invoice"
    assert "confidence" in payload["vehicle_vin"]
    assert "review_required" in payload["vehicle_vin"]
    assert payload["vehicle_make"]["value"] == "GENESIS"
    assert payload["vehicle_make"]["review_required"] is True
    assert payload["extra"]["Certificate of Origin for a Vehicle"] == [
        {"field_name": "plant_code", "value": "A1", "confidence": 0.8}
    ]
    assert payload["extra"]["Dealer Invoice"] == [
        {"field_name": "stock_number", "value": "123", "confidence": 0.9}
    ]


def test_consolidate_extractions_handles_split_document_parts(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "bundle"
    extracted_dir = artifact_dir / "extracted"
    extracted_dir.mkdir(parents=True)

    (extracted_dir / "Lease_Agreement.json").write_text(
        json.dumps(
            [
                {
                    "document_type": "motor_vehicle_lease_agreement",
                    "source_document_name": "Motor Vehicle Lease Agreement (part 1/2)",
                    "lessee_name": {"value": "NAFTALI T SHAPIRO", "confidence": 0.7},
                    "extra": [
                        {
                            "field_name": "section",
                            "value": "1",
                            "confidence": 0.9,
                        }
                    ],
                },
                {
                    "document_type": "motor_vehicle_lease_agreement",
                    "source_document_name": "Motor Vehicle Lease Agreement (part 2/2)",
                    "lessee_name": {"value": "NAFTALI T. SHAPIRO", "confidence": 0.8},
                    "extra": [
                        {
                            "field_name": "section",
                            "value": "2",
                            "confidence": 0.9,
                        }
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )

    result = consolidate_extractions(extracted_dir, artifact_dir)
    payload = json.loads(result.output_json.read_text(encoding="utf-8"))

    assert "NAFTALI" in payload["lessee_name"]["value"]
    assert "SHAPIRO" in payload["lessee_name"]["value"]
    assert len(payload["lessee_name"]["variants"]) == 2
    assert payload["lessee_name"]["variants"][0]["source_document_name"] == "Motor Vehicle Lease Agreement (part 2/2)"
    assert "Motor Vehicle Lease Agreement (part 1/2)" in payload["extra"]
    assert "Motor Vehicle Lease Agreement (part 2/2)" in payload["extra"]
