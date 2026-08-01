from pathlib import Path

import pytest

import wsj_pipeline.discovery as discovery_module
from wsj_pipeline.discovery import discover_sources


def create_file(path: Path, content: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_discovers_mixed_year_layouts_in_stable_order(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    create_file(archive / "2016" / "z-story.html")
    create_file(archive / "2024" / "2024" / "a-story.HTML")
    create_file(archive / "2024" / "2024" / "a-story_main_image.WEBP")

    candidates = list(discover_sources(archive))

    assert [item.relative_html_path for item in candidates] == [
        "2016/z-story.html",
        "2024/2024/a-story.HTML",
    ]
    assert candidates[1].relative_header_image_path == (
        "2024/2024/a-story_main_image.WEBP"
    )


def test_excludes_hidden_macos_and_backup_directories(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    create_file(archive / "2016" / "included.html")
    create_file(archive / ".agents" / "hidden.html")
    create_file(archive / "2018" / "__MACOSX" / "artifact.html")
    create_file(archive / "backup" / "copied.html")

    candidates = list(discover_sources(archive))

    assert [item.relative_html_path for item in candidates] == ["2016/included.html"]


def test_excludes_only_top_level_backup_while_retaining_all_depth_rules(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    create_file(archive / "backup" / "top-level-copy.html")
    create_file(archive / "2024" / "backup" / "editorial.html")
    create_file(archive / "2024" / "backup" / ".hidden" / "hidden.html")
    create_file(
        archive / "2024" / "backup" / "__MACOSX" / "artifact.html"
    )

    candidates = list(discover_sources(archive))

    assert [item.relative_html_path for item in candidates] == [
        "2024/backup/editorial.html"
    ]


def test_missing_image_is_valid_and_warned(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    create_file(archive / "2016" / "story.html", b"article")

    [candidate] = list(discover_sources(archive))

    assert candidate.relative_header_image_path is None
    assert candidate.header_image_size is None
    assert candidate.header_image_mtime_ns is None
    assert candidate.warnings == ("missing_image",)


def test_discovery_excludes_html_and_header_image_leaf_symlinks(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    external_html = tmp_path / "external.html"
    external_html.write_bytes(b"external synthetic article")
    linked_html = archive / "2016" / "linked.html"
    linked_html.parent.mkdir(parents=True)
    linked_html.symlink_to(external_html)
    real_html = archive / "2016" / "real.html"
    create_file(real_html, b"real synthetic article")
    external_image = tmp_path / "external.jpg"
    external_image.write_bytes(b"external synthetic image")
    (archive / "2016" / "real_main_image.jpg").symlink_to(external_image)

    candidates = list(discover_sources(archive))

    assert [candidate.relative_html_path for candidate in candidates] == [
        "2016/real.html"
    ]
    assert candidates[0].relative_header_image_path is None
    assert candidates[0].warnings == ("missing_image",)


def test_ambiguous_images_use_extension_priority_and_warn(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    create_file(archive / "2016" / "story.html")
    create_file(archive / "2016" / "story_main_image.webp", b"webp")
    create_file(archive / "2016" / "story_main_image.jpg", b"jpeg")

    [candidate] = list(discover_sources(archive))

    assert candidate.relative_header_image_path == "2016/story_main_image.jpg"
    assert candidate.header_image_size == 4
    assert candidate.warnings == ("multiple_images",)


def test_limit_is_applied_after_global_sorting(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    for relative_path in ("2020/c.html", "2016/a.html", "2019/b.html"):
        create_file(archive / relative_path)

    candidates = list(discover_sources(archive, limit=2))

    assert [item.relative_html_path for item in candidates] == [
        "2016/a.html",
        "2019/b.html",
    ]


def test_global_lexical_order_merges_files_and_directories(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    for relative_path in (
        "a/story.html",
        "a.html",
        "a-early.html",
        "z.html",
    ):
        create_file(archive / relative_path)

    candidates = list(discover_sources(archive))

    assert [item.relative_html_path for item in candidates] == [
        "a-early.html",
        "a.html",
        "a/story.html",
        "z.html",
    ]


def test_limited_discovery_yields_before_traversing_later_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "archive"
    create_file(archive / "a" / "first.html")
    blocked_directory = archive / "z"
    create_file(blocked_directory / "later.html")
    real_scandir = discovery_module.os.scandir

    def guarded_scandir(path):
        if Path(path) == blocked_directory:
            raise AssertionError("limited discovery traversed beyond its first yield")
        return real_scandir(path)

    monkeypatch.setattr(discovery_module.os, "scandir", guarded_scandir)

    iterator = discover_sources(archive, limit=1)

    assert next(iterator).relative_html_path == "a/first.html"


def test_candidate_records_html_stat_fingerprint(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    html_path = archive / "2016" / "story.html"
    create_file(html_path, b"article bytes")

    [candidate] = list(discover_sources(archive))
    stat = html_path.stat()

    assert candidate.html_size == len(b"article bytes")
    assert candidate.html_mtime_ns == stat.st_mtime_ns


def test_image_pairing_scans_each_source_directory_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    archive = tmp_path / "archive"
    folder = archive / "2016"
    for index in range(4):
        create_file(folder / f"story-{index}.html")
        create_file(folder / f"story-{index}_main_image.jpg")
    real_index_images = discovery_module._index_images
    scanned_directories: list[Path] = []

    def counted_index_images(path: Path):
        scanned_directories.append(path)
        return real_index_images(path)

    monkeypatch.setattr(discovery_module, "_index_images", counted_index_images)

    assert len(list(discover_sources(archive))) == 4
    assert scanned_directories.count(folder) == 1
