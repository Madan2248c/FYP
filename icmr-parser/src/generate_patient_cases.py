#!/usr/bin/env python3
"""
Generate synthetic patient cases from ICMR syndrome data.

Creates realistic patient cases with correct and incorrect prescriptions
for training the reasoning LLM.

Usage:
    python src/generate_patient_cases.py --output data/patient_cases_large.json --count 100
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class PatientCaseGenerator:
    """Generate synthetic patient cases from ICMR data."""
    
    def __init__(self, icmr_data_path: str):
        """Load ICMR syndrome data."""
        with open(icmr_data_path, 'r', encoding='utf-8') as f:
            self.syndromes = json.load(f)
        
        print(f"✓ Loaded {len(self.syndromes)} syndromes from ICMR data")
        
        # Filter syndromes with usable data
        self.usable_syndromes = [
            s for s in self.syndromes 
            if len(s.get('first_line_drugs', [])) > 0
        ]
        
        print(f"✓ {len(self.usable_syndromes)} syndromes have usable drug data")
    
    def generate_cases(
        self,
        count: int = 50,
        correct_ratio: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Generate patient cases.
        
        Args:
            count: Number of cases to generate
            correct_ratio: Ratio of correct prescriptions (0.6 = 60% correct, 40% incorrect)
        
        Returns:
            List of patient case dictionaries
        """
        cases = []
        
        num_correct = int(count * correct_ratio)
        num_incorrect = count - num_correct
        
        print(f"\nGenerating {count} cases:")
        print(f"  - {num_correct} correct prescriptions ({correct_ratio:.0%})")
        print(f"  - {num_incorrect} incorrect prescriptions ({1-correct_ratio:.0%})")
        
        # Generate correct prescriptions
        for i in range(num_correct):
            syndrome = random.choice(self.usable_syndromes)
            case = self._generate_correct_case(syndrome, i + 1)
            cases.append(case)
        
        # Generate incorrect prescriptions
        for i in range(num_incorrect):
            syndrome = random.choice(self.usable_syndromes)
            case = self._generate_incorrect_case(syndrome, num_correct + i + 1)
            cases.append(case)
        
        # Shuffle to mix correct and incorrect
        random.shuffle(cases)
        
        return cases
    
    def _generate_correct_case(self, syndrome: Dict, case_num: int) -> Dict:
        """Generate a case with correct prescription from ICMR guidelines."""
        
        # Pick a random first-line drug
        drug_info = random.choice(syndrome['first_line_drugs'])
        
        # Generate patient profile
        profile = self._generate_patient_profile(syndrome)
        
        # Generate case ID
        syndrome_abbr = self._get_syndrome_abbrev(syndrome['syndrome_name'])
        case_id = f"{syndrome_abbr}_{case_num:03d}"
        
        return {
            'case_id': case_id,
            'patient_profile': profile,
            'diagnosis': syndrome['syndrome_name'],
            'symptoms': self._generate_symptoms(syndrome),
            'prescription': {
                'drug': drug_info['drug_name'],
                'dosage': drug_info.get('dosage', 'Not specified'),
                'route': drug_info.get('route', 'Not specified'),
                'frequency': drug_info.get('frequency', 'Not specified'),
                'duration': drug_info.get('duration', 'Not specified')
            },
            'expected_outcome': 'Approved',  # Ground truth
            'note': 'Correct prescription following ICMR 2025 guidelines'
        }
    
    def _generate_incorrect_case(self, syndrome: Dict, case_num: int) -> Dict:
        """Generate a case with deliberate errors."""
        
        # Pick a drug to prescribe
        if random.random() < 0.7 and syndrome['first_line_drugs']:
            # 70% chance: Use correct drug but wrong dosage/duration
            drug_info = random.choice(syndrome['first_line_drugs'])
            error_type = random.choice(['underdose', 'overdose', 'wrong_duration', 'wrong_route'])
            
            if error_type == 'underdose':
                # Reduce dosage significantly
                dosage = self._reduce_dosage(drug_info.get('dosage', '500mg'))
                expected = 'Modify'
                note = 'Underdosed - needs higher dose'
            elif error_type == 'overdose':
                # Increase dosage significantly
                dosage = self._increase_dosage(drug_info.get('dosage', '500mg'))
                expected = 'Modify'
                note = 'Overdosed - needs lower dose'
            elif error_type == 'wrong_duration':
                # Reduce duration too much
                dosage = drug_info.get('dosage', '500mg')
                duration = '1-2 days'  # Too short
                expected = 'Modify'
                note = 'Duration too short'
            else:  # wrong_route
                dosage = drug_info.get('dosage', '500mg')
                duration = drug_info.get('duration', '5-7 days')
                route = 'IM' if drug_info.get('route') == 'IV' else 'IV'
                expected = 'Modify'
                note = 'Wrong route of administration'
            
            prescription = {
                'drug': drug_info['drug_name'],
                'dosage': dosage if error_type in ['underdose', 'overdose', 'wrong_duration'] else drug_info.get('dosage', '500mg'),
                'route': route if error_type == 'wrong_route' else drug_info.get('route', 'PO'),
                'frequency': drug_info.get('frequency', 'twice daily'),
                'duration': duration if error_type == 'wrong_duration' else drug_info.get('duration', '5-7 days')
            }
        
        else:
            # 30% chance: Use wrong drug entirely
            wrong_drugs = ['Ciprofloxacin', 'Metronidazole', 'Gentamicin', 'Vancomycin', 'Ampicillin']
            prescription = {
                'drug': random.choice(wrong_drugs),
                'dosage': '500mg',
                'route': 'PO',
                'frequency': 'twice daily',
                'duration': '5 days'
            }
            expected = 'Reject'
            note = 'Wrong drug - not in ICMR recommendations'
        
        # Generate patient profile
        profile = self._generate_patient_profile(syndrome)
        
        # Generate case ID
        syndrome_abbr = self._get_syndrome_abbrev(syndrome['syndrome_name'])
        case_id = f"{syndrome_abbr}_ERR_{case_num:03d}"
        
        return {
            'case_id': case_id,
            'patient_profile': profile,
            'diagnosis': syndrome['syndrome_name'],
            'symptoms': self._generate_symptoms(syndrome),
            'prescription': prescription,
            'expected_outcome': expected,  # Ground truth
            'note': note
        }
    
    def _generate_patient_profile(self, syndrome: Dict) -> Dict:
        """Generate realistic patient profile."""
        age = random.randint(18, 80)
        sex = random.choice(['M', 'F'])
        
        # Common comorbidities
        possible_history = [
            'diabetes', 'hypertension', 'asthma', 'COPD',
            'heart failure', 'chronic kidney disease', 'obesity'
        ]
        
        # 30% chance of having comorbidities
        history = []
        if random.random() < 0.3:
            num_comorbidities = random.randint(1, 2)
            history = random.sample(possible_history, num_comorbidities)
        
        # 10% chance of allergies
        allergies = []
        if random.random() < 0.1:
            allergies = [random.choice(['penicillin', 'sulfa', 'fluoroquinolones'])]
        
        return {
            'age': age,
            'sex': sex,
            'history': history,
            'allergies': allergies
        }
    
    def _generate_symptoms(self, syndrome: Dict) -> List[str]:
        """Generate relevant symptoms for syndrome."""
        
        # Symptom mappings for common syndromes
        symptom_map = {
            'pneumonia': ['fever', 'cough', 'dyspnea', 'chest pain', 'productive sputum'],
            'uti': ['dysuria', 'frequency', 'urgency', 'suprapubic pain', 'hematuria'],
            'urinary tract': ['dysuria', 'frequency', 'urgency', 'suprapubic pain'],
            'cellulitis': ['erythema', 'swelling', 'warmth', 'tenderness'],
            'skin': ['erythema', 'swelling', 'pain', 'purulent drainage'],
            'meningitis': ['fever', 'headache', 'neck stiffness', 'photophobia', 'altered mental status'],
            'gastroenteritis': ['diarrhea', 'vomiting', 'abdominal pain', 'fever'],
            'pharyngitis': ['sore throat', 'fever', 'difficulty swallowing', 'tonsillar exudate'],
            'otitis': ['ear pain', 'fever', 'hearing loss', 'ear discharge'],
            'sinusitis': ['facial pain', 'nasal congestion', 'purulent discharge', 'fever']
        }
        
        syndrome_name = syndrome['syndrome_name'].lower()
        
        for key, symptoms in symptom_map.items():
            if key in syndrome_name:
                # Return 3-5 random symptoms
                num_symptoms = random.randint(3, min(5, len(symptoms)))
                return random.sample(symptoms, num_symptoms)
        
        # Default symptoms
        return ['fever', 'malaise', 'localized pain']
    
    def _get_syndrome_abbrev(self, name: str) -> str:
        """Get abbreviation for syndrome name."""
        abbrev_map = {
            'Community Acquired Pneumonia': 'CAP',
            'Hospital Acquired Pneumonia': 'HAP',
            'Urinary Tract Infections': 'UTI',
            'Skin and Soft Tissue Infections': 'SSTI',
            'Meningitis': 'MEN',
            'Sepsis': 'SEP',
            'Gastroenteritis': 'GE',
            'Pharyngitis': 'PHAR',
            'Otitis Media': 'OM',
            'Sinusitis': 'SIN'
        }
        
        # Try exact match
        if name in abbrev_map:
            return abbrev_map[name]
        
        # Generate from first letters
        words = name.split()
        return ''.join(w[0].upper() for w in words[:3])
    
    def _reduce_dosage(self, dosage_str: str) -> str:
        """Reduce dosage by 50-75%."""
        try:
            # Extract number
            import re
            match = re.search(r'(\d+)', dosage_str)
            if match:
                dose = int(match.group(1))
                reduced = int(dose * random.uniform(0.25, 0.5))
                return dosage_str.replace(str(dose), str(reduced))
        except:
            pass
        return '250mg'  # Fallback
    
    def _increase_dosage(self, dosage_str: str) -> str:
        """Increase dosage by 150-200%."""
        try:
            # Extract number
            import re
            match = re.search(r'(\d+)', dosage_str)
            if match:
                dose = int(match.group(1))
                increased = int(dose * random.uniform(1.5, 2.0))
                return dosage_str.replace(str(dose), str(increased))
        except:
            pass
        return '2g'  # Fallback


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic patient cases from ICMR data'
    )
    parser.add_argument(
        '--icmr',
        type=str,
        default='data/structured/all_syndromes_gemini.json',
        help='Path to ICMR syndrome data'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/patient_cases_generated.json',
        help='Output file for generated cases'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=50,
        help='Number of cases to generate (default: 50)'
    )
    parser.add_argument(
        '--correct-ratio',
        type=float,
        default=0.6,
        help='Ratio of correct prescriptions (default: 0.6 = 60%%)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("PATIENT CASE GENERATOR")
    print("="*60)
    
    # Initialize generator
    generator = PatientCaseGenerator(args.icmr)
    
    # Generate cases
    cases = generator.generate_cases(
        count=args.count,
        correct_ratio=args.correct_ratio
    )
    
    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Generated {len(cases)} cases")
    print(f"✅ Saved to: {args.output}")
    
    # Print statistics
    correct_count = sum(1 for c in cases if c['expected_outcome'] == 'Approved')
    modify_count = sum(1 for c in cases if c['expected_outcome'] == 'Modify')
    reject_count = sum(1 for c in cases if c['expected_outcome'] == 'Reject')
    
    print(f"\n📊 Distribution:")
    print(f"  Approved: {correct_count} ({correct_count/len(cases):.0%})")
    print(f"  Modify: {modify_count} ({modify_count/len(cases):.0%})")
    print(f"  Reject: {reject_count} ({reject_count/len(cases):.0%})")
    
    # Show sample
    print(f"\n📝 Sample cases:")
    for i, case in enumerate(cases[:3], 1):
        print(f"\n{i}. {case['case_id']}")
        print(f"   Diagnosis: {case['diagnosis']}")
        print(f"   Drug: {case['prescription']['drug']}")
        print(f"   Expected: {case['expected_outcome']}")
        print(f"   Note: {case['note']}")


if __name__ == '__main__':
    main()

