from dmv.preprocess.modes import PreprocessMode, preprocess_mode_for_document_type


def test_preprocess_mode_for_driver_license() -> None:
    assert preprocess_mode_for_document_type("driver_license") is PreprocessMode.EMBEDDED_CARD


def test_preprocess_mode_for_insurance_card() -> None:
    assert preprocess_mode_for_document_type("insurance_card") is PreprocessMode.DEFAULT


def test_preprocess_mode_for_dealer_invoice() -> None:
    assert preprocess_mode_for_document_type("dealer_invoice") is PreprocessMode.FULL_PAGE_FORM


def test_preprocess_mode_for_unknown_type() -> None:
    assert preprocess_mode_for_document_type("other") is PreprocessMode.DEFAULT
