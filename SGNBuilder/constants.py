
import pkgutil
import json

census_files2fields = json.loads(pkgutil.get_data(__name__, "static/census_files2fields.json"))
census_fields2files = json.loads(pkgutil.get_data(__name__, "static/census_fields2files.json"))
census_desc2field = json.loads(pkgutil.get_data(__name__, "static/census_desc2field.json"))

state_county_ids = json.loads(pkgutil.get_data(__name__, "static/state_county_ids.json"))
state_ids = json.loads(pkgutil.get_data(__name__, "static/state_ids.json"))
