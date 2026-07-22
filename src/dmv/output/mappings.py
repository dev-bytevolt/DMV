from __future__ import annotations

import re
from typing import Any

from dmv.output.formatting import (
    format_currency_field,
    format_odometer_reading,
    normalize_vin,
    today_form_date,
)
from dmv.output.tax_resolution import (
    compute_lfis_amount,
    epa_rating,
    gross_vehicle_price,
)
from dmv.output.values import first_value, get_consolidated_value, truthy_flag

# Blank filenames under artifacts/blanks/
BLANK_UTA = "OS-SS-UTA - BLANK 2024.pdf"
BLANK_BA49 = "BA-49 BLANK-2022.pdf"
BLANK_OWNERSHIP = "NEW CAR OWNERSHIP - BLANK.pdf"

# Written under artifact_dir/output/
OUTPUT_COVER = "Cover_Letter.pdf"
OUTPUT_UTA = "Universal_Title_Application.pdf"
OUTPUT_BA49 = "Application_for_Vehicle_Registration.pdf"
OUTPUT_OWNERSHIP = "New_Car_Ownership.pdf"

OUTPUT_PACKET_FILENAME = "output.pdf"

# Ownership radio groups: /0 = Yes (left), /1 = No (right).
_RADIO_YES = "/0"
_RADIO_NO = "/1"


def _put(fields: dict[str, str], name: str, value: str | None) -> None:
    if value:
        fields[name] = value


def _looks_like_entity(name: str) -> bool:
    upper = name.upper()
    return any(
        token in upper
        for token in ("TRUST", "LLC", "INC", "LTD", "LEASE", "TITLING", "CORP", "COMPANY")
    )


def requests_new_plates(data: dict[str, Any]) -> bool:
    """True when plate number fields should stay empty (NEW PLATES packets)."""
    plate_type = (first_value(data, "plate_type") or "").strip().upper()
    if plate_type and "NEW" in plate_type and "PLATE" in plate_type:
        return True
    if plate_type and "PLATE" in plate_type and "NEW" in plate_type:
        return True
    # Cover letters default to NEW PLATES when a retail plate exists or none does.
    return not plate_type or "NEW PLATE" in plate_type


# MV Express agent defaults for UTA Step 5 (Representative Information).
# Telephone intentionally left blank per MV Express practice on filled samples.
_MV_EXPRESS_REPRESENTATIVE: dict[str, str] = {
    "First Name": "DINA",
    "Last Name": "NAMDAR",
    "Address_2": "160 EMPIRE BLVD",
    "CityTown_3": "BROOKLYN",
    "State_3": "NY",
    "Zip Code_3": "11225",
}


def _apply_mv_express_representative(fields: dict[str, str]) -> None:
    """Fill blank Step 5 representative fields with MV Express agent defaults."""
    for name, value in _MV_EXPRESS_REPRESENTATIVE.items():
        if not fields.get(name):
            fields[name] = value


def build_uta_fields(data: dict[str, Any]) -> dict[str, str]:
    fields: dict[str, str] = {}
    vin = normalize_vin(first_value(data, "vehicle_vin")) or first_value(
        data, "vehicle_vin"
    )
    _put(fields, "Vehicle Identification Number VIN", vin)
    # Blank AcroForm names are swapped vs labels: Vehicle Year/Make = Year_2/Make_2.
    _put(fields, "Year_2", first_value(data, "vehicle_year"))
    _put(fields, "Make_2", first_value(data, "vehicle_make"))
    _put(fields, "Model", first_value(data, "vehicle_model"))
    # Fuel Type widget sits in the Vessel column — leave empty for vehicles.
    if not requests_new_plates(data):
        _put(fields, "NJ License Plate Number", first_value(data, "plate_number"))
    _put(fields, "Color", first_value(data, "vehicle_color"))
    _put(
        fields,
        "Weight",
        first_value(data, "vehicle_weight", "vehicle_weight_or_passengers"),
    )
    body = first_value(data, "vehicle_body_type") or ""
    # Odometer disclosures often include a short body code like "WAGON 4 DO".
    if body.upper().startswith("WAGON"):
        body = "WAGON 4" if "WAGON 4" in body.upper() or body.upper().startswith("WAGON") else body
        if body.upper().startswith("WAGON") and "4" in body.upper():
            body = "WAGON 4"
    _put(fields, "Body Type", body)
    _put(
        fields,
        "Odometer Reading at time of purchase",
        format_odometer_reading(first_value(data, "odometer_reading")),
    )

    owner_name = first_value(
        data, "owner.full_name", "owner.name", "owner_full_name", "owner_name", "buyer_name"
    )
    _put(fields, "Owner Full Name or Entity Name", owner_name)
    _put(fields, "Telephone Number", first_value(data, "owner.phone", "owner_phone"))
    owner_id_paths = [
        "owner.license_or_entity_id",
        "owner_license_or_entity_id",
    ]
    # Driver license IDs belong to the lessee/driver, not a lease titling trust.
    if not (owner_name and _looks_like_entity(owner_name)):
        owner_id_paths.extend(["driver.license_number", "driver_license_number"])
    _put(
        fields,
        "Driver License or MVC Business Entity Identification Number",
        first_value(data, *owner_id_paths),
    )
    _put(
        fields,
        "Address",
        first_value(
            data,
            "owner.address.street",
            "owner_address_street",
            "driver.address.street",
            "driver_address_street",
        ),
    )
    _put(
        fields,
        "CityTown",
        first_value(
            data,
            "owner.address.city",
            "owner_address_city",
            "driver.address.city",
            "driver_address_city",
        ),
    )
    _put(
        fields,
        "State",
        first_value(
            data,
            "owner.address.state",
            "owner_address_state",
            "driver.address.state",
            "driver_address_state",
        ),
    )
    _put(
        fields,
        "Zip Code",
        first_value(
            data,
            "owner.address.zip",
            "owner_address_zip",
            "driver.address.zip",
            "driver_address_zip",
        ),
    )

    _put(
        fields,
        "CoOwner First Name if applicable",
        first_value(data, "co_owner_first_name"),
    )
    _put(
        fields,
        "CoOwner Last Name if applicable",
        first_value(data, "co_owner_last_name"),
    )
    _put(
        fields,
        "CoOwner Driver License Number if applicable",
        first_value(data, "co_owner_license_number"),
    )

    _put(
        fields,
        "Lienholder Name",
        first_value(data, "lienholder.name", "lien_holder", "lienholder_name"),
    )
    _put(
        fields,
        "Driver License or MVC Business Entity Identification Number_2",
        first_value(data, "lienholder.id", "lienholder_id"),
    )
    _put(
        fields,
        "Telephone Number_2",
        first_value(data, "lienholder.phone", "lienholder_phone"),
    )
    _put(
        fields,
        "Lienholder Address",
        first_value(data, "lienholder.address.street", "lienholder_address_street"),
    )
    _put(
        fields,
        "CityTown_2",
        first_value(data, "lienholder.address.city", "lienholder_address_city"),
    )
    _put(
        fields,
        "State_2",
        first_value(data, "lienholder.address.state", "lienholder_address_state"),
    )
    _put(
        fields,
        "Zip Code_2",
        first_value(data, "lienholder.address.zip", "lienholder_address_zip"),
    )

    _put(
        fields,
        "First Name",
        first_value(data, "representative.first_name", "representative_first_name"),
    )
    _put(
        fields,
        "Last Name",
        first_value(data, "representative.last_name", "representative_last_name"),
    )
    _put(
        fields,
        "Telephone Number_3",
        first_value(data, "representative.phone", "representative_phone"),
    )
    _put(
        fields,
        "Address_2",
        first_value(
            data,
            "representative.address.street",
            "representative_address_street",
        ),
    )
    _put(
        fields,
        "CityTown_3",
        first_value(data, "representative.address.city", "representative_address_city"),
    )
    _put(
        fields,
        "State_3",
        first_value(data, "representative.address.state", "representative_address_state"),
    )
    _put(
        fields,
        "Zip Code_3",
        first_value(data, "representative.address.zip", "representative_address_zip"),
    )
    _apply_mv_express_representative(fields)

    if truthy_flag(get_consolidated_value(data, "odometer_not_actual")) is True:
        fields["N  Not actual mileage"] = "/Yes"
    if truthy_flag(get_consolidated_value(data, "odometer_exceeded_mechanical")) is True:
        fields["M  Mileage has exceeded mechanical limitations"] = "/Yes"

    return fields


def _format_lessee_name(raw: str | None) -> str | None:
    if not raw:
        return None
    cleaned = " ".join(raw.replace(",", " ").split())
    # Keep source ordering when already "LAST, FIRST"
    if "," in (raw or ""):
        return cleaned.upper()
    parts = cleaned.split()
    if len(parts) < 2:
        return cleaned.upper()
    return f"{parts[-1].upper()}, {' '.join(p.upper() for p in parts[:-1])}"


def _split_license_segments(raw: str | None) -> tuple[str, str, str] | None:
    """NJ DL / corpcode is printed as three 5-character segments on BA-49."""
    if not raw:
        return None
    digits = re.sub(r"[^A-Za-z0-9]", "", raw.upper())
    if len(digits) < 10:
        parts = raw.split()
        if len(parts) >= 3:
            return parts[0][:5], parts[1][:5], parts[2][:5]
        return None
    return digits[0:5], digits[5:10], digits[10:15]


def _split_dob_parts(raw: str | None) -> tuple[str, str, str] | None:
    if not raw:
        return None
    match = re.search(
        r"(\d{1,2})\D+(\d{1,2})\D+(\d{2,4})",
        raw.strip(),
    )
    if not match:
        return None
    month, day, year = match.group(1), match.group(2), match.group(3)
    if len(year) == 2:
        year = f"19{year}" if int(year) > 30 else f"20{year}"
    return month.zfill(2) if len(month) == 1 else month, day.zfill(2) if len(day) == 1 else day, year


def _put_ba49_person_row(
    fields: dict[str, str],
    *,
    row: int,
    license_number: str | None = None,
    gender: str | None = None,
    eye_color: str | None = None,
    dob: str | None = None,
    ssn: str | None = None,
) -> None:
    """Fill owner (0) / co-owner (1) / lessee (2) credential row on BA-49."""
    segments = _split_license_segments(license_number)
    if segments:
        _put(fields, f"Text24.0.{row}", segments[0])
        _put(fields, f"Text24.1.{row}", segments[1])
        _put(fields, f"Text24.2.{row}", segments[2])
    _put(fields, f"Gender.{row}", (gender or "").strip().upper()[:1] or None)
    eyes = (eye_color or "").strip().upper()
    if eyes in {"BROWN", "BR"}:
        eyes = "BRN"
    elif eyes in {"BLUE", "BL"}:
        eyes = "BLU"
    elif eyes in {"HAZEL", "HZ"}:
        eyes = "HZL"
    elif eyes in {"GREEN", "GR"}:
        eyes = "GRN"
    _put(fields, f"Eye Color.{row}", eyes or None)
    dob_parts = _split_dob_parts(dob)
    if dob_parts:
        _put(fields, f"Text27.0.{row}", dob_parts[0])
        _put(fields, f"Text27.1.{row}", dob_parts[1])
        _put(fields, f"Text27.2.{row}", dob_parts[2])
    ssn_digits = re.sub(r"\D", "", ssn or "")
    if len(ssn_digits) >= 9:
        _put(fields, f"Text29.0.{row}", ssn_digits[0:3])
        _put(fields, f"Text29.1.{row}", ssn_digits[3:5])
        _put(fields, f"Text29.2.{row}", ssn_digits[5:9])


def build_ba49_fields(data: dict[str, Any]) -> dict[str, str]:
    fields: dict[str, str] = {}
    if not requests_new_plates(data):
        _put(fields, "Plate Number", first_value(data, "plate_number"))
        _put(fields, "Prefix", first_value(data, "plate_prefix"))
    vin = normalize_vin(first_value(data, "vehicle_vin")) or first_value(
        data, "vehicle_vin"
    )
    _put(fields, "Vehicle Identification Number (VIN)", vin)
    owner_name = first_value(
        data,
        "owner.full_name",
        "owner.name",
        "owner_full_name",
        "owner_name",
        "buyer_name",
    )
    _put(fields, "Name/Owner", owner_name)
    lessee_raw = first_value(data, "lessee.name", "lessee_name")
    # Keep natural "FIRST LAST" order from source docs (BA-49 samples vary).
    _put(fields, "Name/Lessee", lessee_raw)
    _put(
        fields,
        "Street Address",
        first_value(
            data,
            "owner.address.street",
            "owner_address_street",
            "driver.address.street",
            "driver_address_street",
        ),
    )
    _put(
        fields,
        "Street Address_2",
        first_value(
            data,
            "lessee.address.street",
            "lessee_address_street",
            "driver.address.street",
            "driver_address_street",
        ),
    )
    _put(
        fields,
        "City",
        first_value(
            data,
            "owner.address.city",
            "owner_address_city",
            "driver.address.city",
            "driver_address_city",
        ),
    )
    _put(
        fields,
        "State",
        first_value(
            data,
            "owner.address.state",
            "owner_address_state",
            "driver.address.state",
            "driver_address_state",
        ),
    )
    _put(
        fields,
        "Zip",
        first_value(
            data,
            "owner.address.zip",
            "owner_address_zip",
            "driver.address.zip",
            "driver_address_zip",
        ),
    )
    _put(fields, "County", first_value(data, "owner.county", "owner_county"))
    _put(
        fields,
        "City_2",
        first_value(
            data,
            "lessee.address.city",
            "lessee_address_city",
            "driver.address.city",
            "driver_address_city",
        ),
    )
    _put(
        fields,
        "State_2",
        first_value(
            data,
            "lessee.address.state",
            "lessee_address_state",
            "driver.address.state",
            "driver_address_state",
        ),
    )
    _put(
        fields,
        "Zip_2",
        first_value(
            data,
            "lessee.address.zip",
            "lessee_address_zip",
            "driver.address.zip",
            "driver_address_zip",
        ),
    )
    _put(
        fields,
        "Date Lease Signed",
        first_value(data, "lease_signed_date", "lease_start_date"),
    )
    _put(fields, "Term (Months)", first_value(data, "lease_term_months"))
    _put(
        fields,
        "Name/Co-Owner",
        first_value(data, "co_owner_name", "co_owner_first_name"),
    )
    _put(fields, "Requested Registration Code", first_value(data, "registration_code"))
    weight_or_passengers = first_value(data, "vehicle_weight_or_passengers")
    if not weight_or_passengers:
        weight_or_passengers = first_value(data, "vehicle_weight")
    _put(fields, "Weight or Number of Passengers", weight_or_passengers)
    _put(fields, "Date Lease Cancelled", first_value(data, "lease_cancelled_date"))
    _put(fields, "Insurance Company", first_value(data, "insurance_company"))
    _put(fields, "Policy Number", first_value(data, "insurance_policy_number"))

    # Registration type + default No answers used on filled samples.
    fields["Check Box23.1"] = "/Initial"
    fields["Check Box23.7.1.0"] = "/No"
    fields["Check Box30.0.0"] = "/No"
    fields["Check Box30.0.1"] = "/No"

    _put(
        fields,
        "Text31.1.1",
        format_odometer_reading(first_value(data, "odometer_reading")),
    )

    # Owner row: corpcode / entity id when the titled owner is an entity.
    _put_ba49_person_row(
        fields,
        row=0,
        license_number=first_value(
            data, "owner.license_or_entity_id", "owner_license_or_entity_id"
        ),
    )
    # Lessee / driver credential row.
    _put_ba49_person_row(
        fields,
        row=2,
        license_number=first_value(
            data, "driver.license_number", "driver_license_number"
        ),
        gender=first_value(data, "driver.gender", "driver_gender"),
        eye_color=first_value(data, "driver.eyes_color", "driver_eyes_color"),
        dob=first_value(data, "driver.dob", "driver_dob"),
        ssn=first_value(data, "driver.ssn", "driver_ssn", "lessee.ssn"),
    )
    return fields


def _sane_epa_rating(raw: str | None) -> str | None:
    """Reject curb-weight mistaken as EPA (typical EPA is 10–150)."""
    if not raw:
        return None
    cleaned = raw.strip()
    try:
        value = float(cleaned.replace(",", ""))
    except ValueError:
        return cleaned
    if value > 150:
        return None
    # Prefer integer display when whole number.
    if value == int(value):
        return str(int(value))
    return cleaned


def _ownership_yes_no(
    *,
    yes: bool,
) -> str:
    return _RADIO_YES if yes else _RADIO_NO


def build_ownership_fields(data: dict[str, Any]) -> dict[str, str]:
    fields: dict[str, str] = {}
    vin = normalize_vin(first_value(data, "vehicle_vin")) or first_value(
        data, "vehicle_vin"
    )
    _put(fields, "Vehicle Identification Number", vin)
    epa = _sane_epa_rating(first_value(data, "vehicle_epa_mpg_rating"))
    _put(
        fields,
        "ModelList the Average EPA miles per gallon rating Add both city and highway ratings and divide by 2 OR designate as Not Rated and skip to Step 4",
        epa,
    )
    _put(
        fields,
        "New Vehicle Dealership Name",
        first_value(data, "dealership.name", "dealership_name", "dealer_name"),
    )
    _put(
        fields,
        "Business Entity CorpCode Number",
        first_value(data, "dealership.entity_id", "dealership_entity_id"),
    )
    _put(
        fields,
        "Address",
        first_value(data, "dealership.address.street", "dealership_address_street"),
    )
    _put(
        fields,
        "City  Town",
        first_value(data, "dealership.address.city", "dealership_address_city"),
    )
    _put(
        fields,
        "State",
        first_value(data, "dealership.address.state", "dealership_address_state"),
    )
    _put(
        fields,
        "Zip Code",
        first_value(data, "dealership.address.zip", "dealership_address_zip"),
    )
    _put(fields, "Date", today_form_date())
    _put(fields, "Make", first_value(data, "vehicle_make"))
    _put(fields, "Model", first_value(data, "vehicle_model"))
    gross = gross_vehicle_price(data)
    if gross is not None:
        _put(fields, "Grose", format_currency_field(gross))
    else:
        _put(
            fields,
            "Grose",
            format_currency_field(
                first_value(data, "gross_sales_lease_price", "purchase_price")
            ),
        )
    lfis = compute_lfis_amount(data)
    if lfis is not None:
        _put(fields, "Surcharge", format_currency_field(lfis))
    else:
        _put(
            fields,
            "Surcharge",
            format_currency_field(first_value(data, "surcharge_amount", "lfis_amount")),
        )

    # Left-margin step checkmarks (shared field name, all widgets).
    fields["Check Box2"] = "/Yes"

    epa_val = epa_rating(data)
    gross_val = gross_vehicle_price(data)
    fuel = (first_value(data, "vehicle_fuel_type") or "").upper()
    zero_emission = any(
        token in fuel for token in ("ELECTRIC", "EV", "ZEV", "BEV", "FUEL CELL")
    )
    out_of_state = truthy_flag(get_consolidated_value(data, "titled_outside_nj")) is True
    commercial = truthy_flag(get_consolidated_value(data, "commercial_vehicle")) is True

    # Step 3 questions (Group1–4): Yes only when the statutory condition holds.
    fields["Group1"] = _ownership_yes_no(
        yes=epa_val is not None
        and epa_val > 40
        and gross_val is not None
        and gross_val >= 45000
    )
    fields["Group2"] = _ownership_yes_no(yes=zero_emission)
    fields["Group3"] = _ownership_yes_no(yes=commercial)
    fields["Group4"] = _ownership_yes_no(yes=out_of_state)

    # Step 4 LFIS triggers.
    fields["Group6"] = _ownership_yes_no(
        yes=epa_val is not None
        and epa_val < 40
        and gross_val is not None
        and gross_val >= 45000
    )
    fields["Group7"] = _ownership_yes_no(
        yes=epa_val is not None
        and epa_val < 19
        and (gross_val is None or gross_val < 45000)
    )
    return fields
