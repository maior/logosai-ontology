"""업로드 경로 보존 — 폴더 정보를 문 앞에서 버리지 않는다.

**종전 동작의 두 가지 손실** (`save_dataset` 의 `Path(filename).name`):
  1. 폴더 계층이 사라진다 — "제안서/본문.pdf" → "본문.pdf". 폴더 업로드가
     의미를 가지려면 상대 경로가 남아야 한다.
  2. **이름 충돌 시 조용히 덮어쓴다** — "a/보고서.pdf" 와 "b/보고서.pdf" 를
     같이 올리면 뒤가 앞을 지운다. 데이터 유실인데 아무도 모른다.

**설계**: 저장 이름은 상대 경로를 평탄화(`/` → `__`)해 traversal 없이 한 디렉터리에
담고, 원 상대 경로는 **매니페스트**(데이터셋 디렉터리 *밖*)에 남긴다 — 안에 두면
list/analyze 가 매니페스트를 업로드 파일로 오인해 인제스트한다.
"""
import json

import pytest


@pytest.fixture()
def svc(tmp_path):
    from ontology.server.service import OntologyBuilderService
    return OntologyBuilderService(data_dir=tmp_path)


class TestPathPreservation:
    def test_folder_path_flattened_and_recorded(self, svc, tmp_path):
        res = svc.save_dataset([("제안서/부속/본문.pdf", b"x")])
        assert res["files"] == ["제안서__부속__본문.pdf"]
        manifest = svc.dataset_manifest(res["dataset_id"])
        assert manifest["제안서__부속__본문.pdf"] == "제안서/부속/본문.pdf"

    def test_collision_no_longer_overwrites(self, svc):
        """**종전의 조용한 데이터 유실** — a/x 와 b/x 가 이제 둘 다 남는다."""
        res = svc.save_dataset([("a/보고서.pdf", b"AAA"), ("b/보고서.pdf", b"BBB")])
        assert len(res["files"]) == 2
        assert len(set(res["files"])) == 2
        dataset_dir = svc.dataset_path(res["dataset_id"])
        contents = {p.read_bytes() for p in dataset_dir.iterdir()}
        assert contents == {b"AAA", b"BBB"}

    def test_identical_rel_path_twice_gets_suffix(self, svc):
        res = svc.save_dataset([("x.txt", b"1"), ("x.txt", b"2")])
        assert len(set(res["files"])) == 2

    def test_traversal_is_neutralized(self, svc):
        """`..` 는 경로 세그먼트에서 제거된다 — 데이터셋 디렉터리를 벗어날 수 없다."""
        res = svc.save_dataset([("../../etc/passwd", b"x")])
        dataset_dir = svc.dataset_path(res["dataset_id"])
        saved = list(dataset_dir.iterdir())
        assert len(saved) == 1
        assert dataset_dir in saved[0].parents
        assert ".." not in saved[0].name
        # 매니페스트도 정화된 경로만 담는다 — 원문 보존이 traversal 문자열
        # 보존을 뜻하면 매니페스트가 공격 문자열의 저장소가 된다.
        manifest = svc.dataset_manifest(res["dataset_id"])
        assert all(".." not in v for v in manifest.values())

    def test_flat_filename_unchanged(self, svc):
        """폴더 없는 업로드는 종전과 완전히 같아야 한다 — 하위호환 관문
        (chunk_id 가 source 문자열에 의존하므로 이름이 바뀌면 캐시가 무효화된다)."""
        res = svc.save_dataset([("약관.pdf", b"x")])
        assert res["files"] == ["약관.pdf"]

    def test_manifest_lives_outside_the_dataset_dir(self, svc):
        """안에 두면 list_datasets/analyze 가 업로드 파일로 오인해 인제스트한다."""
        res = svc.save_dataset([("폴더/문서.txt", b"x")])
        dataset_dir = svc.dataset_path(res["dataset_id"])
        assert not any("manifest" in p.name for p in dataset_dir.iterdir())
        listed = [d for d in svc.list_datasets()
                  if d["dataset_id"] == res["dataset_id"]][0]
        assert all("manifest" not in f for f in listed["files"])

    def test_manifest_missing_is_empty_not_error(self, svc):
        """옛 데이터셋(매니페스트 없던 시절)도 조회는 성립해야 한다."""
        assert svc.dataset_manifest("ds_nonexistent") == {}
