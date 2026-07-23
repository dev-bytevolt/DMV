from __future__ import annotations

# Canonical field names derived from artifacts/blanks output forms and common
# source documents. Use empty string when a field is not present in the source.
CANONICAL_EXTRACTION_FIELDS: tuple[str, ...] = (
    # Cover letter (COVER SHEET DMV.docx)
    "cover_date",
    "customer_name",
    "vehicle_vin",
    "lien_holder",
    "plate_type",
    "vehicle_color",
    "purchase_price",
    "sales_tax",
    "lfis_amount",
    "collect_taxes",
    "collect_lfis",
    # Universal Title Application (OS-SS-UTA)
    "vehicle_year",
    "vehicle_make",
    "vehicle_model",
    "vehicle_fuel_type",
    "vehicle_body_type",
    "vehicle_weight",
    "vehicle_axles",
    "odometer_reading",
    "odometer_not_actual",
    "odometer_exceeded_mechanical",
    "owner_full_name",
    "owner_phone",
    "owner_license_or_entity_id",
    "owner_address_street",
    "owner_address_city",
    "owner_address_state",
    "owner_address_zip",
    "co_owner_first_name",
    "co_owner_last_name",
    "co_owner_license_number",
    "lienholder_name",
    "lienholder_id",
    "lienholder_phone",
    "lienholder_address_street",
    "lienholder_address_city",
    "lienholder_address_state",
    "lienholder_address_zip",
    "representative_first_name",
    "representative_last_name",
    "representative_phone",
    "representative_address_street",
    "representative_address_city",
    "representative_address_state",
    "representative_address_zip",
    # Dealership Supplemental Title Form (NEW CAR OWNERSHIP)
    "vehicle_epa_mpg_rating",
    "dealership_name",
    "dealership_entity_id",
    "dealership_address_street",
    "dealership_address_city",
    "dealership_address_state",
    "dealership_address_zip",
    "gross_sales_lease_price",
    "surcharge_amount",
    # Application for Vehicle Registration (BA-49)
    "plate_number",
    "plate_prefix",
    "owner_name",
    "lessee_name",
    "co_owner_name",
    "owner_county",
    "lease_signed_date",
    "lease_term_months",
    "lease_cancelled_date",
    "registration_code",
    "vehicle_weight_or_passengers",
    "insurance_company",
    "insurance_policy_number",
    "lessee_address_street",
    "lessee_address_city",
    "lessee_address_state",
    "lessee_address_zip",
    # Driver license
    "driver_first_name",
    "driver_last_name",
    "driver_middle_name",
    "driver_full_name",
    "driver_address_street",
    "driver_address_city",
    "driver_address_state",
    "driver_address_zip",
    "driver_license_number",
    "driver_dob",
    "driver_gender",
    "driver_height",
    "driver_eyes_color",
    "driver_ssn",
    "driver_license_issue_date",
    "driver_license_expiration_date",
    "driver_license_class",
    "driver_endorsements",
    "driver_restrictions",
    # Insurance
    "insured_name",
    "policy_effective_date",
    "policy_expiration_date",
    # Dealer invoice / retail certificate of sale
    "dealer_name",
    "buyer_name",
    "seller_name",
    "sale_date",
    "sales_tax_amount",
    "sales_tax_paid",
    "sales_tax_exemption_code",
    # Lease agreement
    "lessor_name",
    "monthly_payment",
    "lease_start_date",
    "lease_end_date",
    # Check payment
    "payee_name",
    "check_amount",
    "check_date",
    "check_number",
    "check_memo",
    # Tax / entity identifiers
    "federal_tax_id",
    "federal_tax_identification_number",
    # Power of attorney
    "poa_grantor_name",
    "poa_agent_name",
    "poa_date",
    # Manufacturer certificate / odometer disclosure
    "certificate_number",
    "manufacturer_name",
    # General
    "entity_name",
    "document_date",
    "phone_number",
    "email_address",
)

FIELD_DESCRIPTIONS: dict[str, str] = {
    "cover_date": "Cover letter date",
    "customer_name": "Customer or applicant name on cover letter",
    "vehicle_vin": "Vehicle Identification Number (17 characters)",
    "lien_holder": "Lienholder name",
    "plate_type": (
        "Plate request wording when present (e.g. NEW PLATES, transfer). "
        "Do not put the plate number here — use plate_number for that"
    ),
    "vehicle_color": (
        "Vehicle color as a full color name when printed (e.g. GRAY, BLACK, WHITE, "
        "CELESTIAL GRAY). Keep a color code only if that is all that appears"
    ),
    "purchase_price": (
        "On invoices, prefer the amount sales tax was computed on when identifiable; "
        "otherwise the vehicle selling/retail price line (e.g. NEW CAR RETAIL). "
        "Not total cash price including fees. On leases, capitalized cost / amount "
        "subject to tax when shown. Capture total cash separately via "
        "gross_sales_lease_price when needed"
    ),
    "sales_tax": (
        "Sales tax amount and/or wording such as TAX SATISFIED / PAID when tax was "
        "already remitted by the dealer — include that wording when shown instead of "
        "or in addition to a numeric amount"
    ),
    "lfis_amount": (
        "Luxury/fuel inefficient vehicle surcharge (LFIS) when shown "
        "(often ~0.4% of gross). Same concept as surcharge_amount when both appear"
    ),
    "collect_taxes": "Whether NJ DMV should collect taxes (YES/NO)",
    "collect_lfis": "Whether NJ DMV should collect LFIS (YES/NO)",
    "sales_tax_paid": (
        "Whether sales tax was already paid/satisfied by the dealer (YES/NO)"
    ),
    "sales_tax_exemption_code": "Sales tax exemption code if present",
    "vehicle_year": "Vehicle model year",
    "vehicle_make": "Vehicle make",
    "vehicle_model": (
        "Vehicle model name without unnecessary trim/package unless the trim is part "
        "of the model designation on the source (e.g. RAV4, not RAV4 XLE AWD)"
    ),
    "vehicle_fuel_type": "Vehicle fuel type",
    "vehicle_body_type": (
        "Standard body style only (WAGON, SEDAN, SUV, PICKUP, COUPE, VAN, etc.). "
        "Never trim, series, or package names like XLE, AWD, PLUG-IN HYBRID"
    ),
    "vehicle_weight": "Vehicle weight",
    "vehicle_axles": "Number of axles (passenger vehicles are typically 2)",
    "odometer_reading": "Odometer reading at time of sale or disclosure",
    "odometer_not_actual": "Odometer not actual mileage indicator (Y/N)",
    "odometer_exceeded_mechanical": "Odometer exceeded mechanical limits (Y/N)",
    "owner_full_name": "Owner full name or entity name",
    "owner_phone": (
        "Telephone number for the titled owner/lessor when present "
        "(not the lessee/customer unless they are the titled owner)"
    ),
    "owner_license_or_entity_id": (
        "Driver license or MVC business entity ID for the titled owner/lessor "
        "when present"
    ),
    "owner_address_street": "Owner street address",
    "owner_address_city": "Owner city",
    "owner_address_state": "Owner state",
    "owner_address_zip": "Owner ZIP code",
    "co_owner_first_name": "Co-owner first name",
    "co_owner_last_name": "Co-owner last name",
    "co_owner_license_number": "Co-owner driver license number",
    "lienholder_name": "Lienholder name",
    "lienholder_id": "Lienholder driver license or entity ID",
    "lienholder_phone": "Lienholder telephone number",
    "lienholder_address_street": "Lienholder street address",
    "lienholder_address_city": "Lienholder city",
    "lienholder_address_state": "Lienholder state",
    "lienholder_address_zip": "Lienholder ZIP code",
    "representative_first_name": "Representative first name",
    "representative_last_name": "Representative last name",
    "representative_phone": "Representative telephone number",
    "representative_address_street": "Representative street address",
    "representative_address_city": "Representative city",
    "representative_address_state": "Representative state",
    "representative_address_zip": "Representative ZIP code",
    "vehicle_epa_mpg_rating": (
        "Average EPA miles per gallon rating when printed on the document"
    ),
    "dealership_name": "Dealership name",
    "dealership_entity_id": (
        "Numeric MVC/facility business entity number (e.g. 7104407). "
        "Not a state dealer code like NY704"
    ),
    "dealership_address_street": "Dealership street address",
    "dealership_address_city": "Dealership city",
    "dealership_address_state": "Dealership state",
    "dealership_address_zip": "Dealership ZIP code",
    "gross_sales_lease_price": (
        "Gross sales or lease price of the vehicle used for LFIS/surcharge "
        "calculation (e.g. NEW CAR RETAIL / vehicle price before add-ons when shown). "
        "Not necessarily total cash price including fees"
    ),
    "surcharge_amount": (
        "Luxury/fuel inefficient vehicle surcharge when shown "
        "(often ~0.4% of gross). Same concept as lfis_amount when both appear"
    ),
    "plate_number": "License plate number",
    "plate_prefix": "License plate prefix",
    "owner_name": "Registered owner name",
    "lessee_name": "Lessee name",
    "co_owner_name": "Co-owner name",
    "owner_county": "Owner county",
    "lease_signed_date": "Date lease was signed",
    "lease_term_months": "Lease term in months",
    "lease_cancelled_date": "Date lease was cancelled",
    "registration_code": "Requested registration code",
    "vehicle_weight_or_passengers": "Vehicle weight or number of passengers",
    "insurance_company": "Insurance company name",
    "insurance_policy_number": "Insurance policy number",
    "lessee_address_street": "Lessee street address",
    "lessee_address_city": "Lessee city",
    "lessee_address_state": "Lessee state",
    "lessee_address_zip": "Lessee ZIP code",
    "driver_first_name": "Driver first name",
    "driver_last_name": "Driver last name",
    "driver_middle_name": "Driver middle name",
    "driver_full_name": "Driver full name as printed on license",
    "driver_address_street": "Driver street address",
    "driver_address_city": "Driver city",
    "driver_address_state": "Driver state",
    "driver_address_zip": "Driver ZIP code",
    "driver_license_number": "Driver license number",
    "driver_dob": "Driver date of birth",
    "driver_gender": "Driver gender",
    "driver_height": "Driver height",
    "driver_eyes_color": "Driver eye color",
    "driver_ssn": "Driver / registrant Social Security number when present",
    "driver_license_issue_date": "Driver license issue date",
    "driver_license_expiration_date": "Driver license expiration date",
    "driver_license_class": "Driver license class",
    "driver_endorsements": "Driver license endorsements",
    "driver_restrictions": "Driver license restrictions",
    "insured_name": "Insured person or entity name",
    "policy_effective_date": "Insurance policy effective date",
    "policy_expiration_date": "Insurance policy expiration date",
    "dealer_name": "Dealer name",
    "buyer_name": "Buyer or purchaser name",
    "seller_name": "Seller name",
    "sale_date": "Sale or transaction date",
    "sales_tax_amount": (
        "Sales tax amount on invoice or receipt, and/or TAX SATISFIED / PAID wording "
        "when shown"
    ),
    "lessor_name": "Lessor name on lease agreement",
    "monthly_payment": "Monthly lease payment",
    "lease_start_date": "Lease start date",
    "lease_end_date": "Lease end date",
    "payee_name": "Check payee name",
    "check_amount": "Check amount",
    "check_date": "Check date",
    "check_number": "Check number",
    "check_memo": "Check memo line",
    "federal_tax_id": "Federal tax ID / EIN when present",
    "federal_tax_identification_number": "Federal tax identification number (EIN)",
    "poa_grantor_name": "Power of attorney grantor name",
    "poa_agent_name": "Power of attorney agent or attorney-in-fact name",
    "poa_date": "Power of attorney date",
    "certificate_number": "Certificate or title number",
    "manufacturer_name": "Vehicle manufacturer name",
    "entity_name": "Business or entity name",
    "document_date": "Primary document date",
    "phone_number": "General phone number",
    "email_address": "Email address",
}
