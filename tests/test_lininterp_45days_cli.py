"""lininterp_45days: patient given on the command line, or the patient with the most SLH episodes."""
from scripts.counterfactual import lininterp_45days as l45


def test_defaults_mean_automatic_patient_and_configured_offset():
    args = l45.parse_arguments([])
    assert args.patient_id is None                    # None -> resolve_patient_id picks the patient with the most SLH episodes
    assert args.offset == l45.SELECTED_OFFSET_MIN
    assert args.out_dir == l45.FIGURES_OUTPUT_PATH


def test_patient_offset_and_output_folder_can_be_given():
    args = l45.parse_arguments(["--patient_id", "P7", "--offset", "45", "--out_dir", "somewhere"])
    assert (args.patient_id, args.offset, args.out_dir) == ("P7", 45.0, "somewhere")
