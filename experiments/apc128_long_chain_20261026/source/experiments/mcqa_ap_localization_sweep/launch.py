"""Verified signed interpreter bootstrap; no safety-policy changes."""
from diagnostics.ap_thermal_20261021.resume_ap import setup
handles=setup()
from .runner import main
main()
