"""Validate the counts used in project progress entries."""

from pathlib import Path

from scripts.verify_phase1 import case_counts


def test_junit_counts(tmp_path: Path) -> None:
    xml = tmp_path / "report.xml"
    xml.write_text(
        "<testsuites><testsuite><testcase/><testcase><skipped/></testcase>"
        "<testcase><failure/></testcase><testcase><error/></testcase></testsuite></testsuites>",
        encoding="utf-8",
    )
    assert case_counts(xml) == {"passed": 1, "skipped": 1, "failed": 1, "errors": 1}
