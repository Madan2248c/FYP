"""
Manual data entry tool for syndromes with missing information.

Interactive CLI for completing syndrome data that automated extraction couldn't capture.
"""

import json
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_models import SyndromeData, Pathogen, DrugRecommendation
from pydantic import ValidationError

logger = logging.getLogger(__name__)


class ManualEntryTool:
    """Interactive tool for manual syndrome data entry."""
    
    def __init__(self, syndromes_file: str, review_queue_file: str):
        """
        Initialize manual entry tool.
        
        Args:
            syndromes_file: Path to all_syndromes_v2.json
            review_queue_file: Path to manual_review_queue.csv
        """
        self.syndromes_file = syndromes_file
        self.review_queue_file = review_queue_file
        self.syndromes = []
        self.review_queue = []
        self.progress = {}
        self.progress_file = Path(syndromes_file).parent / "manual_entry_progress.json"
        
        # Load data
        self._load_syndromes()
        self._load_review_queue()
        self._load_progress()
    
    def _load_syndromes(self):
        """Load existing syndrome data."""
        try:
            with open(self.syndromes_file, 'r', encoding='utf-8') as f:
                self.syndromes = json.load(f)
            print(f"✓ Loaded {len(self.syndromes)} syndromes")
        except Exception as e:
            print(f"❌ Failed to load syndromes: {e}")
            sys.exit(1)
    
    def _load_review_queue(self):
        """Load manual review queue."""
        try:
            df = pd.read_csv(self.review_queue_file)
            self.review_queue = df.to_dict('records')
            print(f"✓ Loaded {len(self.review_queue)} syndromes needing review")
        except Exception as e:
            print(f"❌ Failed to load review queue: {e}")
            sys.exit(1)
    
    def _load_progress(self):
        """Load progress from previous session."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    self.progress = json.load(f)
                completed = len([v for v in self.progress.values() if v])
                print(f"✓ Resuming: {completed} of {len(self.review_queue)} already completed")
            except Exception as e:
                print(f"⚠️  Could not load progress: {e}")
                self.progress = {}
        else:
            self.progress = {}
    
    def _save_progress(self):
        """Save progress to file."""
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, indent=2)
    
    def _save_syndromes(self):
        """Save updated syndrome data."""
        # Create backup first
        backup_file = Path(self.syndromes_file).with_suffix('.backup.json')
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(self.syndromes, f, indent=2, ensure_ascii=False)
        
        # Save updated file
        with open(self.syndromes_file, 'w', encoding='utf-8') as f:
            json.dump(self.syndromes, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved (backup at {backup_file})")
    
    def _display_syndrome_context(self, syndrome: Dict, review_item: Dict):
        """Display syndrome information for context."""
        print("\n" + "=" * 80)
        print(f"SYNDROME: {syndrome.get('syndrome_name', 'Unknown')}")
        print("=" * 80)
        print(f"Page: {syndrome.get('source_page', 'Unknown')}")
        print(f"Issues: {review_item.get('issues', 'Unknown')}")
        print()
        
        # Show definition
        definition = syndrome.get('definition', '')
        if definition:
            print("DEFINITION:")
            print("-" * 80)
            print(self._wrap_text(definition, 80))
            print()
        
        # Show existing data
        existing_pathogens = syndrome.get('common_pathogens', [])
        if existing_pathogens:
            print(f"EXISTING PATHOGENS: ({len(existing_pathogens)})")
            for p in existing_pathogens[:3]:
                print(f"  - {p.get('organism_name', 'Unknown')}")
            if len(existing_pathogens) > 3:
                print(f"  ... and {len(existing_pathogens) - 3} more")
            print()
        
        existing_drugs = syndrome.get('first_line_drugs', [])
        if existing_drugs:
            print(f"EXISTING DRUGS: ({len(existing_drugs)})")
            for d in existing_drugs[:3]:
                print(f"  - {d.get('drug_name', 'Unknown')} {d.get('dosage', '')}")
            if len(existing_drugs) > 3:
                print(f"  ... and {len(existing_drugs) - 3} more")
            print()
        
        # Show text excerpt
        excerpt = review_item.get('text_excerpt', '')
        if excerpt:
            print("TEXT EXCERPT:")
            print("-" * 80)
            print(self._wrap_text(excerpt, 80))
            print()
    
    def _wrap_text(self, text: str, width: int) -> str:
        """Wrap text to specified width."""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)
    
    def _input_pathogens(self) -> List[Dict]:
        """
        Interactive input for pathogens.
        
        Returns:
            List of pathogen dictionaries
        """
        print("\n" + "-" * 80)
        print("ENTER PATHOGENS")
        print("-" * 80)
        print("Enter organism names (one per line)")
        print("Format: Genus species (e.g., Escherichia coli)")
        print("Type 'done' when finished, 'skip' to skip this section")
        print()
        
        pathogens = []
        count = 1
        
        while True:
            organism = input(f"{count}. Organism name: ").strip()
            
            if organism.lower() == 'done':
                break
            
            if organism.lower() == 'skip':
                return []
            
            if not organism:
                continue
            
            # Ask for resistance pattern
            resistance = input(f"   Resistance pattern (optional, e.g., ESBL, MRSA): ").strip()
            
            pathogen_dict = {
                'organism_name': organism,
                'resistance_pattern': resistance if resistance else None,
                'prevalence': None,
                'source': 'manual_entry'
            }
            
            pathogens.append(pathogen_dict)
            count += 1
            print(f"   ✓ Added: {organism}")
        
        return pathogens
    
    def _input_drugs(self, drug_type: str = "first-line") -> List[Dict]:
        """
        Interactive input for drugs.
        
        Args:
            drug_type: "first-line" or "alternative"
            
        Returns:
            List of drug dictionaries
        """
        print("\n" + "-" * 80)
        print(f"ENTER {drug_type.upper()} DRUGS")
        print("-" * 80)
        print("Enter drug information (one drug at a time)")
        print("Type 'done' when finished, 'skip' to skip this section")
        print()
        
        drugs = []
        count = 1
        
        while True:
            print(f"\nDrug #{count}:")
            drug_name = input("  Drug name (e.g., Ceftriaxone): ").strip()
            
            if drug_name.lower() == 'done':
                break
            
            if drug_name.lower() == 'skip':
                return []
            
            if not drug_name:
                continue
            
            # Ask for details
            dosage = input("  Dosage (e.g., 2g, 50mg/kg): ").strip()
            route = input("  Route (IV/PO/IM/SC): ").strip().upper()
            duration = input("  Duration (e.g., 7-10 days): ").strip()
            frequency = input("  Frequency (optional, e.g., every 12 hours): ").strip()
            
            # Validate
            if not dosage:
                print("  ⚠️  Warning: Dosage is required")
            if not route:
                print("  ⚠️  Warning: Route is required")
            
            drug_dict = {
                'drug_name': drug_name,
                'dosage': dosage if dosage else None,
                'route': route if route else None,
                'duration': duration if duration else None,
                'frequency': frequency if frequency else None,
                'indication': drug_type,
                'source': 'manual_entry'
            }
            
            drugs.append(drug_dict)
            count += 1
            print(f"  ✓ Added: {drug_name} {dosage} {route}")
        
        return drugs
    
    def _review_and_confirm(self, pathogens: List[Dict], first_line: List[Dict], alternative: List[Dict]) -> bool:
        """
        Display entered data and ask for confirmation.
        
        Args:
            pathogens: List of pathogens
            first_line: List of first-line drugs
            alternative: List of alternative drugs
            
        Returns:
            True if confirmed, False to re-enter
        """
        print("\n" + "=" * 80)
        print("REVIEW YOUR ENTRIES")
        print("=" * 80)
        
        print(f"\nPATHOGENS: ({len(pathogens)})")
        for p in pathogens:
            resistance = f" ({p['resistance_pattern']})" if p.get('resistance_pattern') else ""
            print(f"  - {p['organism_name']}{resistance}")
        
        print(f"\nFIRST-LINE DRUGS: ({len(first_line)})")
        for d in first_line:
            print(f"  - {d['drug_name']} {d.get('dosage', '')} {d.get('route', '')} for {d.get('duration', '')}")
        
        if alternative:
            print(f"\nALTERNATIVE DRUGS: ({len(alternative)})")
            for d in alternative:
                print(f"  - {d['drug_name']} {d.get('dosage', '')} {d.get('route', '')} for {d.get('duration', '')}")
        
        print()
        response = input("Is this correct? (yes/no/cancel): ").strip().lower()
        
        if response in ['yes', 'y']:
            return True
        elif response in ['cancel', 'c']:
            sys.exit(0)
        else:
            return False
    
    def process_syndrome(self, index: int, syndrome_idx: int, review_item: Dict) -> bool:
        """
        Process a single syndrome with manual entry.
        
        Args:
            index: Index in review queue
            syndrome_idx: Index in syndromes list
            review_item: Review queue item
            
        Returns:
            True if completed, False if skipped
        """
        syndrome = self.syndromes[syndrome_idx]
        
        # Check if already completed
        if self.progress.get(str(index), False):
            print(f"\n✓ Already completed, skipping...")
            return True
        
        # Display context
        self._display_syndrome_context(syndrome, review_item)
        
        # Ask if user wants to process this syndrome
        action = input("Process this syndrome? (yes/skip/quit): ").strip().lower()
        
        if action in ['quit', 'q']:
            print("\n👋 Quitting. Progress saved.")
            sys.exit(0)
        
        if action in ['skip', 's']:
            print("⏭️  Skipped")
            return False
        
        # Enter data with retry loop
        while True:
            # Input pathogens if needed
            pathogens = []
            if 'pathogen' in review_item.get('issues', '').lower():
                pathogens = self._input_pathogens()
            
            # Input first-line drugs if needed
            first_line = []
            if 'drug' in review_item.get('issues', '').lower():
                first_line = self._input_drugs("first-line")
            
            # Optional: alternative drugs
            if first_line:
                add_alt = input("\nAdd alternative drugs? (yes/no): ").strip().lower()
                alternative = []
                if add_alt in ['yes', 'y']:
                    alternative = self._input_drugs("alternative")
            else:
                alternative = []
            
            # Review and confirm
            if self._review_and_confirm(pathogens, first_line, alternative):
                break
            else:
                print("\n🔄 Let's try again...\n")
        
        # Update syndrome data
        if pathogens:
            # Merge with existing pathogens
            existing = syndrome.get('common_pathogens', [])
            syndrome['common_pathogens'] = existing + pathogens
            print(f"✓ Added {len(pathogens)} pathogens")
        
        if first_line:
            # Merge with existing drugs
            existing = syndrome.get('first_line_drugs', [])
            syndrome['first_line_drugs'] = existing + first_line
            print(f"✓ Added {len(first_line)} first-line drugs")
        
        if alternative:
            # Merge with existing alternative drugs
            existing = syndrome.get('alternative_drugs', [])
            syndrome['alternative_drugs'] = existing + alternative
            print(f"✓ Added {len(alternative)} alternative drugs")
        
        # Mark as manually entered
        syndrome['manually_entered'] = True
        syndrome['manual_entry_date'] = datetime.now().isoformat()
        
        # Update in syndromes list
        self.syndromes[syndrome_idx] = syndrome
        
        # Save progress
        self.progress[str(index)] = True
        self._save_progress()
        self._save_syndromes()
        
        print("\n💾 Saved!")
        return True
    
    def run(self):
        """Run the interactive manual entry session."""
        print("\n" + "=" * 80)
        print("MANUAL SYNDROME DATA ENTRY TOOL")
        print("=" * 80)
        print()
        print("This tool will guide you through entering missing syndrome data.")
        print("Progress is saved after each entry.")
        print("You can quit at any time and resume later.")
        print()
        
        # Calculate progress
        completed = len([v for v in self.progress.values() if v])
        total = len(self.review_queue)
        
        print(f"Progress: {completed}/{total} completed")
        print()
        
        if completed == total:
            print("✅ All syndromes completed!")
            print(f"\nUpdated file: {self.syndromes_file}")
            return
        
        input("Press Enter to start...")
        
        # Process each syndrome
        for idx, review_item in enumerate(self.review_queue):
            # Find syndrome in main list
            syndrome_idx = review_item['index']
            
            print(f"\n{'='*80}")
            print(f"PROGRESS: {idx + 1} of {total}")
            print(f"{'='*80}")
            
            self.process_syndrome(idx, syndrome_idx, review_item)
        
        # Final summary
        print("\n" + "=" * 80)
        print("🎉 ALL SYNDROMES COMPLETED!")
        print("=" * 80)
        print()
        print(f"Updated file: {self.syndromes_file}")
        print(f"Backup saved: {Path(self.syndromes_file).with_suffix('.backup.json')}")
        print()
        
        # Clean up progress file
        if self.progress_file.exists():
            self.progress_file.unlink()
            print("✓ Removed progress file")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Manual data entry tool for syndromes with missing information'
    )
    
    parser.add_argument(
        '--syndromes',
        type=str,
        default='data/structured/all_syndromes_v2.json',
        help='Path to syndromes JSON file'
    )
    
    parser.add_argument(
        '--review-queue',
        type=str,
        default='data/structured/manual_review_queue.csv',
        help='Path to manual review queue CSV'
    )
    
    args = parser.parse_args()
    
    # Check files exist
    if not Path(args.syndromes).exists():
        print(f"❌ Syndromes file not found: {args.syndromes}")
        return 1
    
    if not Path(args.review_queue).exists():
        print(f"❌ Review queue file not found: {args.review_queue}")
        print("\nDid you run the reprocessing script first?")
        print("  python src/reprocess_failed_syndromes.py")
        return 1
    
    # Run tool
    try:
        tool = ManualEntryTool(args.syndromes, args.review_queue)
        tool.run()
        return 0
    except KeyboardInterrupt:
        print("\n\n⏸️  Interrupted. Progress saved. Run again to resume.")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

