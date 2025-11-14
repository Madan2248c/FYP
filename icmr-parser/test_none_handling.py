#!/usr/bin/env python3
"""Test None handling in verifier."""

# Simulate the issue
guideline = {
    'syndrome_name': 'Test Syndrome',
    'first_line_drugs': [{'drug_name': 'Amoxicillin'}],
    'alternative_drugs': None,  # This was causing the error
    'common_pathogens': None
}

# Old way (would fail)
# alternative_drugs = guideline.get('alternative_drugs', [])
# print(f"Old way: {alternative_drugs}")  # Prints: None (not [])

# New way (works)
alternative_drugs = guideline.get('alternative_drugs') or []
common_pathogens = guideline.get('common_pathogens') or []

print("✅ None handling test:")
print(f"  alternative_drugs: {alternative_drugs} (type: {type(alternative_drugs).__name__})")
print(f"  common_pathogens: {common_pathogens} (type: {type(common_pathogens).__name__})")

# Test iteration (should not crash)
try:
    alternative_names = [d.get('drug_name', '').lower() for d in alternative_drugs]
    print(f"  alternative_names: {alternative_names}")
    print("✅ Iteration works!")
except TypeError as e:
    print(f"❌ Error: {e}")

