import hashlib
import json

import pytest

import reproduce


@pytest.fixture
def release(tmp_path):
    (tmp_path / "data.csv").write_text("value\n1\n")
    digest = hashlib.sha256((tmp_path / "data.csv").read_bytes()).hexdigest()
    (tmp_path / "SHA256SUMS").write_text(f"{digest}  data.csv\n")
    return tmp_path


def test_release_checksums():
    assert reproduce.verify_checksums()


def test_input_provenance():
    provenance = json.loads((reproduce.ROOT / "data/provenance.json").read_text())
    for name, record in provenance["inputs"].items():
        assert reproduce.sha(reproduce.ROOT / name) == record["sha256"]


def test_modified_input_is_rejected(release):
    (release / "data.csv").write_text("value\n2\n")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        reproduce.verify_checksums(release)


@pytest.mark.parametrize("name", ["../private", "/absolute", "C:\\private"])
def test_unsafe_checksum_paths_are_rejected(release, name):
    (release / "SHA256SUMS").write_text(f"{'0' * 64}  {name}\n")
    with pytest.raises(ValueError, match="Invalid checksum entry"):
        reproduce.read_hashes(release)


def test_duplicate_checksum_is_rejected(release):
    path = release / "SHA256SUMS"
    path.write_text(path.read_text() * 2)
    with pytest.raises(ValueError, match="Invalid checksum entry"):
        reproduce.read_hashes(release)


@pytest.mark.parametrize("relative", [".", "..", "data", "reference", "plots"])
def test_release_output_is_rejected(release, relative):
    with pytest.raises(ValueError, match="release files"):
        reproduce.validate_output(release / relative, release)


def test_symlink_output_is_rejected(release):
    destination = release / "out/figures"
    destination.mkdir(parents=True)
    try:
        (destination / reproduce.NUMERICAL[0]).symlink_to(release / "data.csv")
    except OSError as error:
        if getattr(error, "winerror", None) == 1314:
            pytest.skip("Symlink permission required")
        raise
    with pytest.raises(ValueError, match="release files"):
        reproduce.validate_output(release / "out", release)


def test_roundoff_is_accepted(release):
    actual = release / "actual.csv"
    actual.write_text("value\n1.000000000000001\n")
    assert reproduce.compare_table(release / "data.csv", actual) < 1e-12


@pytest.mark.parametrize("text", ["value\n2\n", "value\n1\n2\n", "other\n1\n"])
def test_changed_table_is_rejected(release, text):
    actual = release / "actual.csv"
    actual.write_text(text)
    with pytest.raises(ValueError):
        reproduce.compare_table(release / "data.csv", actual)


def test_eight_figures():
    assert len(reproduce.NUMERICAL) == 6
    assert len(reproduce.STATIC) == 2
    for name in (*reproduce.NUMERICAL, *reproduce.STATIC):
        assert (reproduce.ROOT / "reference/figures" / name).is_file()


def test_routing_events():
    from plots import check_events, routing

    rows = routing.read_results(reproduce.ROOT / "data/routing/results.csv")
    protocol = json.loads((reproduce.ROOT / "data/provenance.json").read_text())[
        "routing_protocol"
    ]
    assert len(rows) == 180
    assert check_events(rows, protocol["events"]) == {
        "relaxed": 151,
        "primary": 85,
        "strict": 3,
    }
    rows[0]["event_primary"] = 1 - rows[0]["event_primary"]
    with pytest.raises(ValueError, match="Event mismatch"):
        check_events(rows, protocol["events"])


def test_wilson_bounds():
    from plots.routing import wilson_interval

    assert wilson_interval(0, 12)[0] == pytest.approx(0, abs=1e-15)
    assert wilson_interval(12, 12)[1] == pytest.approx(1, abs=1e-15)
    lo, hi = wilson_interval(6, 12)
    assert lo < 0.5 < hi
