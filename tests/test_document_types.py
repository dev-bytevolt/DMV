from dmv.document_types import DOCUMENT_TYPE_IDS, build_classification_prompt
from dmv.providers.schemas import CLASSIFICATION_JSON_SCHEMA


def test_document_type_ids_include_other() -> None:
    assert "driver_license" in DOCUMENT_TYPE_IDS
    assert "other" in DOCUMENT_TYPE_IDS


def test_classification_prompt_lists_all_types() -> None:
    prompt = build_classification_prompt()
    for type_id in DOCUMENT_TYPE_IDS:
        assert type_id in prompt


def test_classification_prompt_describes_page_grouping_rules() -> None:
    prompt = build_classification_prompt()
    assert "never interleave" in prompt
    assert "empty_pages" in prompt
    assert "separate documents" in prompt
    assert "blank and separator pages are NOT documents" in prompt
    assert "Do not split one physical document into multiple entries" in prompt
    assert "HIGHEST PRIORITY" in prompt
    assert prompt.index("HIGHEST PRIORITY") < prompt.index(
        "Do not split one physical document into multiple entries"
    )

def test_classification_schema_matches_document_types() -> None:
    doc_schema = CLASSIFICATION_JSON_SCHEMA["properties"]["documents"]["items"]
    enum_values = doc_schema["properties"]["type"]["enum"]
    assert enum_values == list(DOCUMENT_TYPE_IDS)
