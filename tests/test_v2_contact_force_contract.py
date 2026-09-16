"""Structural tests for the pure Gate1 object contact-wrench contract."""

from __future__ import annotations

import json
import unittest

from synthesis_v2.contact_force_contract import (
    ContactForceSpec,
    ContactState,
    Decision,
    ObjectBodyBinding,
    ObjectContactWrench,
    evaluate_object_contact_force,
    record_object_contact_wrench,
)


class ContactForceContractTests(unittest.TestCase):
    def binding(self) -> ObjectBodyBinding:
        return ObjectBodyBinding(
            object_id="cube_a",
            body_id=4,
            body_name="cube_a_body",
            geom_ids=(17, 18),
            geom_names=("cube_a_geom", "cube_a_inner_geom"),
            frame="world",
            geom_attribution="body_aggregate",
        )

    def record(self, row, *, sensor_valid=True):
        return record_object_contact_wrench(
            {4: row},
            binding=self.binding(),
            timestamp_s=1.25,
            sensor_valid=sensor_valid,
        )

    def test_no_contact_is_explicit_and_fails_required_contact(self):
        record = self.record((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        result = evaluate_object_contact_force(
            record,
            ContactForceSpec(min_force_n=0.1, require_contact=True),
        )

        self.assertEqual(result.contact_state, ContactState.NO_CONTACT)
        self.assertEqual(result.decision, Decision.FAIL)
        self.assertEqual(result.force_norm_n, 0.0)
        self.assertEqual(result.torque_norm_nm, 0.0)
        self.assertIn("no_contact", result.codes)

    def test_controlled_value_keeps_force_and_torque_separate(self):
        record = self.record((3.0, 4.0, 0.0, 0.0, 0.0, 0.2))
        result = evaluate_object_contact_force(
            record,
            ContactForceSpec(min_force_n=4.0, max_force_n=6.0, require_contact=True),
        )

        self.assertEqual(record.force_n, (3.0, 4.0, 0.0))
        self.assertEqual(record.torque_nm, (0.0, 0.0, 0.2))
        self.assertAlmostEqual(result.force_norm_n, 5.0)
        self.assertAlmostEqual(result.torque_norm_nm, 0.2)
        self.assertEqual(result.decision, Decision.PASS)
        self.assertEqual(result.physical_qualification, Decision.UNKNOWN)
        self.assertNotIn("combined_wrench_norm", record.as_dict())

    def test_mixed_force_torque_norm_is_rejected(self):
        # A caller bypassing the recorder still cannot pass a six-dimensional
        # wrench as ``force_n``.  The contract returns unknown with a specific
        # issue rather than calculating a meaningless mixed-unit norm.
        record = ObjectContactWrench(
            timestamp_s=1.25,
            object_id="cube_a",
            body_id=4,
            body_name="cube_a_body",
            geom_ids=(17,),
            geom_names=("cube_a_geom",),
            frame="world",
            geom_attribution="body_aggregate",
            force_n=(3.0, 4.0, 0.0, 100.0, 0.0, 0.0),
            torque_nm=(0.0, 0.0, 0.2),
            sensor_valid=True,
            source="fixture",
        )
        result = evaluate_object_contact_force(
            record,
            ContactForceSpec(max_force_n=1000.0),
        )

        self.assertEqual(result.decision, Decision.UNKNOWN)
        self.assertTrue(result.has_code("mixed_wrench_not_force"))
        self.assertIsNone(result.force_norm_n)

    def test_missing_sensor_validity_is_unknown_even_with_numeric_row(self):
        record = self.record((1.0, 0.0, 0.0, 0.0, 0.0, 0.0), sensor_valid=None)
        result = evaluate_object_contact_force(
            record,
            ContactForceSpec(max_force_n=2.0),
        )

        self.assertEqual(result.decision, Decision.UNKNOWN)
        self.assertEqual(result.contact_state, ContactState.UNKNOWN)
        self.assertIn("missing_sensor_validity", result.codes)

    def test_missing_row_is_unknown_and_not_zero_filled(self):
        record = record_object_contact_wrench(
            {9: (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)},
            binding=self.binding(),
            timestamp_s=1.25,
            sensor_valid=True,
        )
        result = evaluate_object_contact_force(record, ContactForceSpec(max_force_n=2.0))

        self.assertEqual(result.decision, Decision.UNKNOWN)
        self.assertIn("missing_body_row", result.codes)
        self.assertIsNone(result.force_norm_n)
        self.assertIsNone(record.force_norm_n)

    def test_missing_object_geom_mapping_is_unknown(self):
        record = record_object_contact_wrench(
            {4: (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)},
            binding={
                "object_id": "cube_a",
                "body_id": 4,
                "body_name": "cube_a_body",
                "geom_ids": (),
                "geom_names": (),
                "frame": "world",
                "geom_attribution": "body_aggregate",
            },
            timestamp_s=1.25,
            sensor_valid=True,
        )
        result = evaluate_object_contact_force(record, ContactForceSpec(max_force_n=2.0))

        self.assertEqual(result.decision, Decision.UNKNOWN)
        self.assertIn("missing_geom_ids", result.codes)

    def test_empty_force_spec_is_unknown(self):
        record = self.record((1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        result = evaluate_object_contact_force(record, ContactForceSpec())
        self.assertEqual(result.decision, Decision.UNKNOWN)
        self.assertIn("no_force_constraint", result.codes)

    def test_json_representation_is_split_and_serializable(self):
        record = self.record((3.0, 4.0, 0.0, 0.0, 0.0, 0.2))
        result = evaluate_object_contact_force(record, ContactForceSpec(max_force_n=6.0))
        payload = {"record": record.as_dict(), "evaluation": result.as_dict()}
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        self.assertIn('"unit": "N"', encoded)
        self.assertIn('"unit": "N*m"', encoded)
        self.assertNotIn("combined_wrench_norm", encoded)


if __name__ == "__main__":
    unittest.main()
