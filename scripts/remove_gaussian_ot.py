#!/usr/bin/env python3
"""
Remove gaussian_ot entries from all output JSON files.

This script cleans up the outputs directory by removing gaussian_ot
configurations from all JSON files.

Usage:
    python remove_gaussian_ot.py --dry-run  # Preview changes
    python remove_gaussian_ot.py            # Actually remove entries
"""

import json
import argparse
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple


def is_gaussian_config(key: str, value: Any) -> bool:
    """Check if a config entry is a gaussian_ot variant."""
    # Check by key name
    if key == 'gaussian_ot':
        return True
    if key.startswith('gaussian_'):
        return True
    
    # Check by config content
    if isinstance(value, dict):
        # Direct check
        if value.get('ot_method') == 'gaussian':
            return True
        if value.get('config_name') == 'gaussian_ot':
            return True
        # Check nested metrics
        metrics = value.get('metrics', {})
        if isinstance(metrics, dict):
            if metrics.get('ot_method') == 'gaussian':
                return True
            if metrics.get('config_name') == 'gaussian_ot':
                return True
    
    return False


def clean_configs_list(configs: List[Dict]) -> Tuple[List[Dict], List[str]]:
    """Remove gaussian_ot entries from a list of configs."""
    removed_names = []
    cleaned = []
    
    for c in configs:
        config_name = c.get('config_name', '')
        if config_name == 'gaussian_ot' or c.get('ot_method') == 'gaussian':
            removed_names.append(config_name or 'unnamed')
        else:
            cleaned.append(c)
    
    return cleaned, removed_names


def clean_configs_dict(configs: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Remove gaussian_ot entries from a dict of configs."""
    removed_names = []
    keys_to_remove = []
    
    for key, value in configs.items():
        if is_gaussian_config(key, value):
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        del configs[key]
        removed_names.append(key)
    
    return configs, removed_names


def process_json_file(filepath: Path, dry_run: bool = True) -> Tuple[int, bool]:
    """Process a single JSON file and remove gaussian_ot entries.
    
    Returns (number_removed, was_modified).
    """
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            data = json.loads(content)
    except (json.JSONDecodeError, IOError) as e:
        print(f"  ERROR reading: {e}")
        return 0, False
    
    total_removed = 0
    modified = False
    removed_names = []
    
    # Handle different JSON structures
    
    # 1. configs as list: {"configs": [...]}
    if 'configs' in data and isinstance(data['configs'], list):
        data['configs'], names = clean_configs_list(data['configs'])
        if names:
            removed_names.extend(names)
            total_removed += len(names)
            modified = True
    
    # 2. configs as dict: {"configs": {...}}
    elif 'configs' in data and isinstance(data['configs'], dict):
        data['configs'], names = clean_configs_dict(data['configs'])
        if names:
            removed_names.extend(names)
            total_removed += len(names)
            modified = True
    
    # 3. Top-level check for single config files
    if data.get('config_name') == 'gaussian_ot' or data.get('ot_method') == 'gaussian':
        print(f"  WARNING: This entire file is a gaussian_ot config!")
        # Don't delete the file, just report it
    
    # 4. Clean comparison dict
    if 'comparison' in data and isinstance(data['comparison'], dict):
        keys_to_remove = []
        for metric, value in data['comparison'].items():
            if isinstance(value, dict) and value.get('config') == 'gaussian_ot':
                keys_to_remove.append(metric)
        for key in keys_to_remove:
            del data['comparison'][key]
            modified = True
            print(f"  Removed gaussian_ot reference from comparison[{key}]")
    
    # 5. Clean best_per_metric
    if 'best_per_metric' in data and isinstance(data['best_per_metric'], dict):
        for metric, info in data['best_per_metric'].items():
            if isinstance(info, dict):
                if info.get('overall_best', {}).get('config') == 'gaussian_ot':
                    info['overall_best'] = {'config': None, 'value': None}
                    modified = True
                    print(f"  Cleared gaussian_ot from best_per_metric[{metric}].overall_best")
                if info.get('best_kffm', {}).get('config') == 'gaussian_ot':
                    info['best_kffm'] = {'config': None, 'value': None}
                    modified = True
                    print(f"  Cleared gaussian_ot from best_per_metric[{metric}].best_kffm")
    
    # 6. Update total_configs if present
    if modified and 'total_configs' in data:
        if isinstance(data.get('configs'), list):
            data['total_configs'] = len(data['configs'])
        elif isinstance(data.get('configs'), dict):
            data['total_configs'] = len(data['configs'])
    
    # Report what was removed
    if removed_names:
        print(f"  Removed {len(removed_names)} gaussian configs: {removed_names}")
    
    # Write back if modified
    if modified:
        if dry_run:
            print(f"  [DRY RUN] Would save changes")
        else:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"  SAVED: {filepath}")
    
    return total_removed, modified


def find_all_json_files(outputs_dir: Path) -> List[Path]:
    """Find ALL JSON files in outputs directory recursively."""
    files = []
    
    for root, dirs, filenames in os.walk(outputs_dir):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for filename in filenames:
            if filename.endswith('.json'):
                files.append(Path(root) / filename)
    
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Remove gaussian_ot from output JSON files")
    parser.add_argument('--outputs-dir', type=str, default='../outputs',
                        help='Base outputs directory')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview changes without modifying files')
    parser.add_argument('--file', type=str, default=None,
                        help='Process a specific file instead of all files')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  Remove gaussian_ot from Output JSON Files")
    print("=" * 70)
    
    if args.dry_run:
        print("\n*** DRY RUN MODE - No files will be modified ***")
        print("*** Run without --dry-run to actually apply changes ***\n")
    else:
        print("\n*** LIVE MODE - Files WILL be modified ***\n")
    
    if args.file:
        files = [Path(args.file)]
        if not files[0].exists():
            print(f"ERROR: File not found: {args.file}")
            return
    else:
        outputs_dir = Path(args.outputs_dir)
        if not outputs_dir.exists():
            print(f"ERROR: Directory not found: {outputs_dir}")
            return
        files = find_all_json_files(outputs_dir)
    
    print(f"Found {len(files)} JSON files to process\n")
    print("-" * 70)
    
    total_removed = 0
    files_modified = 0
    
    for filepath in files:
        # Show relative path for cleaner output
        try:
            rel_path = filepath.relative_to(Path(args.outputs_dir).parent)
        except ValueError:
            rel_path = filepath
        
        removed, modified = process_json_file(filepath, dry_run=args.dry_run)
        if removed > 0 or modified:
            print(f"Processing: {rel_path}")
            if removed > 0:
                total_removed += removed
            if modified:
                files_modified += 1
    
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  Files scanned:    {len(files)}")
    print(f"  Files modified:   {files_modified}")
    print(f"  Entries removed:  {total_removed}")
    
    if args.dry_run and (total_removed > 0 or files_modified > 0):
        print("\n  *** Run without --dry-run to apply these changes ***")
    elif not args.dry_run and files_modified > 0:
        print("\n  *** Changes applied successfully ***")
        print("  *** Remember to re-run aggregate_sweep_results.py to update best_per_metric ***")


if __name__ == "__main__":
    main()

