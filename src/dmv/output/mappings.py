from __future__ import annotations

from typing import Any

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


def _put(fields: dict[str, str], name: str, value: str | None) -> None:
    if value:
        fields[name] = value


def build_uta_fields(data: dict[str, Any]) -> dict[str, str]:
    fields: dict[str, str] = {}
    _put(fields, "Vehicle Identification Number VIN", first_value(data, "vehicle_vin"))
    _put(fields, "NJ License Plate Number", first_value(data, "plate_number"))
    _put(fields, "Year", first_value(data, "vehicle_year"))
    _put(fields, "Make", first_value(data, "vehicle_make"))
    _put(fields, "Model", first_value(data, "vehicle_model"))
    _put(fields, "Fuel Type", first_value(data, "vehicle_fuel_type"))
    _put(fields, "Color", first_value(data, "vehicle_color"))
    _put(fields, "Weight", first_value(data, "vehicle_weight", "vehicle_weight_or_passengers"))
    _put(fields, "Body Type", first_value(data, "vehicle_body_type"))
    _put(
        fields,
        "Odometer Reading at time of purchase",
        first_value(data, "odometer_reading"),
    )

    _put(
        fields,
        "Owner Full Name or Entity Name",
        first_value(data, "owner.full_name", "owner.name", "owner_full_name", "owner_name"),
    )
    _put(fields, "Telephone Number", first_value(data, "owner.phone", "owner_phone"))
    _put(
        fields,
        "Driver License or MVC Business Entity Identification Number",
        first_value(data, "owner.license_or_entity_id", "owner_license_or_entity_id"),
    )
    _put(
        fields,
        "Address",
        first_value(data, "owner.address.street", "owner_address_street"),
    )
    _put(fields, "CityTown", first_value(data, "owner.address.city", "owner_address_city"))
    _put(fields, "State", first_value(data, "owner.address.state", "owner_address_state"))
    _put(
        fields,
        "Zip Code",
        first_value(data, "owner.address.zip", "owner_address_zip"),
    )

    _put(fields, "CoOwner First Name if applicable", first_value(data, "co_owner_first_name"))
    _put(fields, "CoOwner Last Name if applicable", first_value(data, "co_owner_last_name"))
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

    if truthy_flag(get_consolidated_value(data, "odometer_not_actual")) is True:
        fields["N  Not actual mileage"] = "/Yes"
    if truthy_flag(get_consolidated_value(data, "odometer_exceeded_mechanical")) is True:
        fields["M  Mileage has exceeded mechanical limitations"] = "/Yes"

    return fields


def build_ba49_fields(data: dict[str, Any]) -> dict[str, str]:
    fields: dict[str, str] = {}
    _put(fields, "Plate Number", first_value(data, "plate_number"))
    _put(fields, "Prefix", first_value(data, "plate_prefix", "plate_type"))
    _put(
        fields,
        "Vehicle Identification Number (VIN)",
        first_value(data, "vehicle_vin"),
    )
    _put(
        fields,
        "Name/Owner",
        first_value(data, "owner.full_name", "owner.name", "owner_full_name", "owner_name"),
    )
    _put(fields, "Name/Lessee", first_value(data, "lessee.name", "lessee_name"))
    _put(
        fields,
        "Street Address",
        first_value(data, "owner.address.street", "owner_address_street"),
    )
    _put(
        fields,
        "Street Address_2",
        first_value(data, "lessee.address.street"),
    )
    _put(fields, "City", first_value(data, "owner.address.city", "owner_address_city"))
    _put(fields, "State", first_value(data, "owner.address.state", "owner_address_state"))
    _put(fields, "Zip", first_value(data, "owner.address.zip", "owner_address_zip"))
    _put(fields, "County", first_value(data, "owner.county", "owner_county"))
    _put(
        fields,
        "City_2",
        first_value(data, "lessee.address.city", "lessee_address_city"),
    )
    _put(
        fields,
        "State_2",
        first_value(data, "lessee.address.state", "lessee_address_state"),
    )
    _put(
        fields,
        "Zip_2",
        first_value(data, "lessee.address.zip", "lessee_address_zip"),
    )
    _put(fields, "Date Lease Signed", first_value(data, "lease_signed_date", "lease_start_date"))
    _put(fields, "Term (Months)", first_value(data, "lease_term_months"))
    _put(
        fields,
        "Name/Co-Owner",
        first_value(data, "co_owner_name", "co_owner_first_name"),
    )
    _put(fields, "Requested Registration Code", first_value(data, "registration_code"))
    _put(
        fields,
        "Weight or Number of Passengers",
        first_value(data, "vehicle_weight_or_passengers", "vehicle_weight"),
    )
    _put(fields, "Date Lease Cancelled", first_value(data, "lease_cancelled_date"))
    _put(
        fields,
        "Insurance Company",
        first_value(data, "insurance_company"),
    )
    _put(fields, "Policy Number", first_value(data, "insurance_policy_number"))
    return fields


def build_ownership_fields(data: dict[str, Any]) -> dict[str, str]:
    fields: dict[str, str] = {}
    _put(
        fields,
        "Vehicle Identification Number",
        first_value(data, "vehicle_vin"),
    )
    _put(
        fields,
        "ModelList the Average EPA miles per gallon rating Add both city and highway ratings and divide by 2 OR designate as Not Rated and skip to Step 4",
        first_value(data, "vehicle_epa_mpg_rating"),
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
    _put(fields, "Date", first_value(data, "sale_date", "cover_date"))
    _put(fields, "Make", first_value(data, "vehicle_make"))
    _put(fields, "Model", first_value(data, "vehicle_model"))
    _put(fields, "Grose", first_value(data, "gross_sales_lease_price", "purchase_price"))
    _put(fields, "Surcharge", first_value(data, "surcharge_amount"))
    return fields
