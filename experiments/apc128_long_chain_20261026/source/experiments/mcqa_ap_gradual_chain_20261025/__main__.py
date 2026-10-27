from diagnostics.ap_thermal_20261021.resume_ap import setup

_dll_handles = setup()
from .runner import main

main()
