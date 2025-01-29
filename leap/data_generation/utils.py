PROVINCE_MAP = {
    "Canada": "CA",
    "British Columbia": "BC",
    "Alberta": "AB",
    "Saskatchewan": "SK",
    "Manitoba": "MB",
    "Ontario": "ON",
    "Quebec": "QC",
    "Newfoundland and Labrador": "NL",
    "Nova Scotia": "NS",
    "New Brunswick": "NB",
    "Prince Edward Island": "PE",
    "Yukon": "YT",
    "Northwest Territories": "NT",
    "Nunavut": "NU"
}


def get_province_id(province: str) -> str:
    return PROVINCE_MAP[province]


def get_sex_id(sex: str) -> str:
    return sex[0:1]


def format_age_group(age_group: str, upper_age_group: str = "100 years and over") -> int:
    if age_group == upper_age_group:
        age = age_group.replace(" years and over", "")
        age = int(age)
    elif age_group == "Under 1 year":
        age = 0
    else:
        age = age_group.replace(" years", "")
        age = age.replace(" year", "")
        age = int(age)
    return age