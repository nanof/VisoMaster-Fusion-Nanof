"""Unit tests for InsightFace GenderAge parse + swap appearance filter."""

from __future__ import annotations

import numpy as np

from app.processors.face_attributes import (
    GENDER_FILTER_ALL,
    GENDER_FILTER_FEMALE,
    GENDER_FILTER_MALE,
    bbox_from_kps5,
    gender_filter_mode,
    parse_gender_age_output,
    skip_swap_for_gender_appearance_filter,
)


def test_parse_gender_age_female_when_idx0_wins():
    gender, conf, age = parse_gender_age_output(np.array([2.0, -2.0, 0.27], dtype=np.float32))
    assert gender == "female"
    assert conf > 0.95
    assert age == 27


def test_parse_gender_age_male_when_idx1_wins():
    gender, conf, age = parse_gender_age_output(np.array([-1.5, 1.5, 0.41], dtype=np.float32))
    assert gender == "male"
    assert conf > 0.9
    assert age == 41


def test_parse_gender_age_matches_insightface_argmax_convention():
    # InsightFace: gender = argmax(pred[:2]); Face.sex = 'M' if gender==1 else 'F'
    gender, _, _ = parse_gender_age_output([0.1, 0.9, 0.3])
    assert gender == "male"
    gender, _, _ = parse_gender_age_output([0.9, 0.1, 0.3])
    assert gender == "female"


def test_skip_filter_all_never_skips():
    assert (
        skip_swap_for_gender_appearance_filter(
            {"GenderAppearanceFilterSelection": GENDER_FILTER_ALL}, "male", 0.99
        )
        is False
    )


def test_skip_female_only_skips_male_when_confident():
    control = {
        "GenderAppearanceFilterSelection": GENDER_FILTER_FEMALE,
        "GenderAppearanceMinConfidenceSlider": 60,
    }
    assert skip_swap_for_gender_appearance_filter(control, "male", 0.80) is True
    assert skip_swap_for_gender_appearance_filter(control, "female", 0.80) is False


def test_skip_male_only_skips_female_when_confident():
    control = {
        "GenderAppearanceFilterSelection": GENDER_FILTER_MALE,
        "GenderAppearanceMinConfidenceSlider": 60,
    }
    assert skip_swap_for_gender_appearance_filter(control, "female", 0.80) is True
    assert skip_swap_for_gender_appearance_filter(control, "male", 0.80) is False


def test_skip_fail_open_on_low_confidence_or_missing():
    control = {
        "GenderAppearanceFilterSelection": GENDER_FILTER_FEMALE,
        "GenderAppearanceMinConfidenceSlider": 70,
    }
    # Low confidence → do not skip (previous brittle behaviour skipped these).
    assert skip_swap_for_gender_appearance_filter(control, "male", 0.50) is False
    assert skip_swap_for_gender_appearance_filter(control, None, 0.99) is False
    assert skip_swap_for_gender_appearance_filter(control, "unknown", 0.99) is False


def test_gender_filter_mode_defaults():
    assert gender_filter_mode(None) == GENDER_FILTER_ALL
    assert gender_filter_mode({}) == GENDER_FILTER_ALL
    assert gender_filter_mode({"GenderAppearanceFilterSelection": "nope"}) == GENDER_FILTER_ALL


def test_bbox_from_kps5_inflates_inner_landmarks():
    kps = np.array(
        [[40.0, 40.0], [60.0, 40.0], [50.0, 55.0], [42.0, 70.0], [58.0, 70.0]],
        dtype=np.float32,
    )
    box = bbox_from_kps5(kps)
    assert box[0] < 40.0
    assert box[1] < 40.0
    assert box[2] > 60.0
    assert box[3] > 70.0
