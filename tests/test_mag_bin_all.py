#!/usr/bin/env python3
"""
Tests for --mag-bin all functionality in malca pipeline.
"""
import pytest
import argparse
from malca.config.config_pipeline import MAG_BINS


def test_mag_bin_all_expansion():
    """Test that 'all' is properly expanded to the full list of magnitude bins."""
    # Simulate the expansion logic from detect.py
    mag_bin = ["all"]
    
    if "all" in mag_bin:
        if len(mag_bin) > 1:
            raise ValueError("Cannot mix 'all' with specific magnitude bins")
        mag_bin = list(reversed(MAG_BINS))
    
    # Verify expansion
    assert mag_bin == list(reversed(MAG_BINS))
    assert len(mag_bin) == 6
    assert mag_bin[0] == "14.5_15"  # First in reverse order
    assert mag_bin[-1] == "12_12.5"  # Last in reverse order


def test_mag_bin_all_mixing_error():
    """Test that mixing 'all' with specific bins raises an error."""
    mag_bin = ["all", "13_13.5"]
    
    with pytest.raises(ValueError, match="Cannot mix 'all' with specific magnitude bins"):
        if "all" in mag_bin and len(mag_bin) > 1:
            raise ValueError("Cannot mix 'all' with specific magnitude bins")


def test_mag_bin_single():
    """Test that single bin specification still works."""
    mag_bin = ["13_13.5"]
    
    # Should not be expanded
    if "all" in mag_bin:
        if len(mag_bin) > 1:
            raise ValueError("Cannot mix 'all' with specific magnitude bins")
        mag_bin = list(reversed(MAG_BINS))
    
    assert mag_bin == ["13_13.5"]
    assert len(mag_bin) == 1


def test_mag_bin_multiple():
    """Test that multiple bin specification still works (backward compatibility)."""
    mag_bin = ["13_13.5", "14_14.5"]
    
    # Should not be expanded
    if "all" in mag_bin:
        if len(mag_bin) > 1:
            raise ValueError("Cannot mix 'all' with specific magnitude bins")
        mag_bin = list(reversed(MAG_BINS))
    
    assert mag_bin == ["13_13.5", "14_14.5"]
    assert len(mag_bin) == 2


def test_mag_bin_tag_generation():
    """Test that mag_bin_tag is correctly generated for different scenarios."""
    # Test single bin
    mag_bin = ["13_13.5"]
    is_auto_all_mode = False
    mag_bin_tag = "all" if is_auto_all_mode else (mag_bin[0] if len(mag_bin) == 1 else "multi")
    assert mag_bin_tag == "13_13.5"
    
    # Test multiple bins
    mag_bin = ["13_13.5", "14_14.5"]
    mag_bin_tag = "all" if is_auto_all_mode else (mag_bin[0] if len(mag_bin) == 1 else "multi")
    assert mag_bin_tag == "multi"
    
    # Test all mode
    mag_bin = list(reversed(MAG_BINS))
    is_auto_all_mode = True
    mag_bin_tag = "all" if is_auto_all_mode else (mag_bin[0] if len(mag_bin) == 1 else "multi")
    assert mag_bin_tag == "all"


def test_mag_bins_config():
    """Test that MAG_BINS is properly configured."""
    assert len(MAG_BINS) == 6
    assert MAG_BINS == ["12_12.5", "12.5_13", "13_13.5", "13.5_14", "14_14.5", "14.5_15"]
    
    # Test reverse order
    reversed_bins = list(reversed(MAG_BINS))
    assert reversed_bins == ["14.5_15", "14_14.5", "13.5_14", "13_13.5", "12.5_13", "12_12.5"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
