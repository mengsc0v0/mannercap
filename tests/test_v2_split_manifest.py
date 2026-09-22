import copy
import unittest

from synthesis_v2.split_manifest import (
    SPLIT_MANIFEST_SCHEMA,
    SplitManifestError,
    manifest_hash,
    validate_split_manifest,
)


def _source(record_id: str, split: str = "development") -> dict:
    return {
        "record_id": record_id,
        "kind": "source_episode",
        "split": split,
        "source_episode_id": record_id,
        "lineage_root_id": "lineage-1",
        "initial_state_id": "state-1",
        "scene_configuration_id": "scene-1",
        "template_family_id": "template-1",
        "pair_id": "pair-1",
    }


class SplitManifestTest(unittest.TestCase):
    def _document(self, records: list[dict]) -> dict:
        document = {"schema": SPLIT_MANIFEST_SCHEMA, "records": records}
        document["manifest_sha256"] = manifest_hash(document)
        return document

    def test_source_and_derived_window_are_valid_in_one_split(self) -> None:
        source = _source("episode-1")
        window = copy.deepcopy(source)
        window.update(
            {
                "record_id": "window-1",
                "kind": "derived_window",
                "source_episode_id": "episode-1",
            }
        )
        audit = validate_split_manifest(self._document([source, window]))
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["record_count"], 2)
        self.assertEqual(audit["split_counts"]["development"], 2)

    def test_related_group_cannot_cross_splits(self) -> None:
        first = _source("episode-1")
        second = _source("episode-2", split="student_test")
        with self.assertRaisesRegex(SplitManifestError, "split leakage"):
            validate_split_manifest(self._document([first, second]))

    def test_derived_window_must_inherit_all_source_groups(self) -> None:
        source = _source("episode-1")
        window = copy.deepcopy(source)
        window.update(
            {
                "record_id": "window-1",
                "kind": "derived_window",
                "source_episode_id": "episode-1",
                "template_family_id": "different-template",
            }
        )
        with self.assertRaisesRegex(SplitManifestError, "must inherit source field"):
            validate_split_manifest(self._document([source, window]))

    def test_embedded_hash_is_checked(self) -> None:
        document = self._document([_source("episode-1")])
        document["manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(SplitManifestError, "manifest_sha256"):
            validate_split_manifest(document)


if __name__ == "__main__":
    unittest.main()
