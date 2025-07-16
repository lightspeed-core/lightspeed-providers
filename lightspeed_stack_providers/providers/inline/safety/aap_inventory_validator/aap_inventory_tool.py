#!/usr/bin/env python3
"""
AAP Inventory Tool

This tool provides validation and comparison functionality for Ansible Automation
Platform inventories (INI format) for containerized and RPM installations
across growth and enterprise topologies.
"""

import argparse
import sys
import configparser
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple


class InventoryProcessor:
    """Base class for processing AAP inventory files."""

    def __init__(self, platform: str = None, topology: str = None):
        self.platform = platform
        self.topology = topology
        self.errors = []
        self.warnings = []

        # Define required sections for each platform/topology combination
        self.required_sections = {
            'containerized': {
                'growth': ['automationgateway', 'automationcontroller', 'automationhub', 'automationeda', 'database'],
                'enterprise': ['automationgateway', 'automationcontroller', 'automationhub', 'automationeda',
                               'execution_nodes', 'redis']
            },
            'rpm': {
                'growth': ['automationgateway', 'automationcontroller', 'execution_nodes', 'automationhub',
                           'automationedacontroller', 'database'],
                'enterprise': ['automationgateway', 'automationcontroller', 'execution_nodes', 'automationhub',
                               'automationedacontroller', 'redis']
            }
        }

        # Define required variables for each platform
        self.required_vars = {
            'containerized': {
                'common': ['postgresql_admin_password'],
                'gateway': ['gateway_admin_password', 'gateway_pg_host', 'gateway_pg_password'],
                'controller': ['controller_admin_password', 'controller_pg_host', 'controller_pg_password'],
                'hub': ['hub_admin_password', 'hub_pg_host', 'hub_pg_password'],
                'eda': ['eda_admin_password', 'eda_pg_host', 'eda_pg_password']
            },
            'rpm': {
                'common': [],
                'gateway': ['automationgateway_admin_password', 'automationgateway_pg_host',
                            'automationgateway_pg_password'],
                'controller': ['admin_password', 'pg_host', 'pg_password'],
                'hub': ['automationhub_admin_password', 'automationhub_pg_host', 'automationhub_pg_password'],
                'eda': ['automationedacontroller_admin_password', 'automationedacontroller_pg_host',
                        'automationedacontroller_pg_password']
            }
        }

    def parse_inventory(self, inventory_path: str) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        """Parse inventory file and return sections and variables."""
        # Check if file exists first
        if not Path(inventory_path).exists():
            raise FileNotFoundError(f"Inventory file not found: {inventory_path}")

        try:
            config = configparser.ConfigParser(allow_no_value=True)
            config.read(inventory_path)
        except Exception as e:
            raise Exception(f"Error parsing inventory file: {e}")

        sections = self._extract_sections(config)
        variables = self._extract_variables(config)

        return sections, variables

    def _extract_sections(self, config: configparser.ConfigParser) -> Dict[str, List[str]]:
        """Extract Ansible group sections from the INI config."""
        sections = {}

        for section_name in config.sections():
            # Skip variable sections
            if section_name.endswith(':vars'):
                continue

            # Extract hosts from the section
            hosts = []
            for key, value in config.items(section_name):
                if value is None or value.strip() == '':
                    # Host without variables
                    hosts.append(key.strip())
                else:
                    # Host with variables
                    hosts.append(f"{key.strip()} {value.strip()}")

            sections[section_name] = hosts

        return sections

    def _extract_variables(self, config: configparser.ConfigParser) -> Dict[str, str]:
        """Extract variables from the [all:vars] section."""
        variables = {}

        if 'all:vars' in config:
            for key, value in config['all:vars'].items():
                variables[key] = value

        return variables

    def get_results(self) -> Dict[str, List[str]]:
        """Get processing results."""
        return {
            'errors': self.errors,
            'warnings': self.warnings
        }


class InventoryValidator(InventoryProcessor):
    """Validates AAP inventory files based on platform and topology."""

    def validate_inventory(self, inventory_path: str) -> bool:
        """Main validation method."""
        try:
            sections, variables = self.parse_inventory(inventory_path)
        except Exception as e:
            self.errors.append(str(e))
            return False

        # Validate sections
        self._validate_sections(sections)

        # Validate variables
        self._validate_variables(variables)

        # Validate topology-specific requirements
        self._validate_topology_requirements(sections)

        return len(self.errors) == 0

    def _validate_sections(self, sections: Dict[str, List[str]]):
        """Validate that all required sections are present."""
        if not self.platform or not self.topology:
            self.warnings.append("Platform and topology not specified, skipping section validation")
            return

        required = self.required_sections[self.platform][self.topology]

        for section in required:
            if section not in sections:
                self.errors.append(f"Missing required section: [{section}]")
            elif not sections[section]:
                self.errors.append(f"Empty required section: [{section}]")

    def _validate_variables(self, variables: Dict[str, str]):
        """Validate that all required variables are present."""
        if not self.platform:
            self.warnings.append("Platform not specified, skipping variable validation")
            return

        required_vars = self.required_vars[self.platform]

        # Check common variables
        for var in required_vars['common']:
            if var not in variables:
                self.errors.append(f"Missing required variable: {var}")

        # Check component-specific variables
        for component in ['gateway', 'controller', 'hub', 'eda']:
            if component in required_vars:
                for var in required_vars[component]:
                    if var not in variables:
                        self.errors.append(f"Missing required {component} variable: {var}")

        # Validate password variables specifically
        self._validate_password_variables(variables)

        # Validate redis_mode settings
        self._validate_redis_mode(variables)

    def _validate_password_variables(self, variables: Dict[str, str]):
        """Validate that password variables are present and not empty."""
        if not self.platform:
            return

        required_vars = self.required_vars[self.platform]
        password_vars = []

        # Collect all password variables from all components
        for component_vars in required_vars.values():
            if isinstance(component_vars, list):
                password_vars.extend([var for var in component_vars if 'password' in var.lower()])

        for var in password_vars:
            if var not in variables:
                self.errors.append(f"Missing required password variable: {var}")
            elif not variables[var] or variables[var].strip() == '':
                self.errors.append(f"Password variable '{var}' is empty")
            elif self._is_variable_placeholder(variables[var]):
                # Allow parameterized variables but warn about them
                self.warnings.append(f"Password variable '{var}' appears to be parameterized: {variables[var]}")

    def _is_variable_placeholder(self, value: str) -> bool:
        """Check if a variable value is a placeholder/template."""
        if not value:
            return False

        # Only consider Jinja2 variables as placeholders
        return '{{' in value

    def _validate_redis_mode(self, variables: Dict[str, str]):
        """Validate redis_mode settings based on topology."""
        if not self.topology:
            return

        if 'redis_mode' in variables:
            redis_mode = variables['redis_mode'].strip()

            if self.topology == 'growth':
                if redis_mode != 'standalone':
                    self.errors.append(f"Growth topology requires redis_mode=standalone, found: {redis_mode}")
            elif self.topology == 'enterprise':
                if redis_mode != 'cluster':
                    self.errors.append(f"Enterprise topology requires redis_mode=cluster, found: {redis_mode}")
        else:
            # redis_mode is required for growth topology
            if self.topology == 'growth':
                self.errors.append("Growth topology requires redis_mode=standalone")

    def _validate_topology_requirements(self, sections: Dict[str, List[str]]):
        """Validate topology-specific requirements."""
        if not self.topology:
            return

        if self.topology == 'growth':
            # Growth topology should have single hosts in most sections
            for section in ['automationgateway', 'automationcontroller', 'automationhub']:
                if section in sections and len(sections[section]) > 1:
                    self.warnings.append(
                        f"Growth topology typically has single host in [{section}], found {len(sections[section])}")

            # For containerized growth, all hostnames/IPs should be the same (all-in-one)
            if self.platform == 'containerized':
                self._validate_containerized_growth_all_in_one(sections)

        elif self.topology == 'enterprise':
            # Enterprise topology requires multiple hosts for HA
            enterprise_sections = ['automationgateway', 'automationcontroller', 'automationhub']

            # For RPM, check automationedacontroller instead of automationeda
            if self.platform == 'rpm':
                enterprise_sections.append('automationedacontroller')
            else:
                enterprise_sections.append('automationeda')

            for section in enterprise_sections:
                if section in sections and len(sections[section]) < 2:
                    self.errors.append(
                        f"Enterprise topology requires at least 2 hosts in [{section}] for HA, found {len(sections[section])}")

    def _validate_containerized_growth_all_in_one(self, sections: Dict[str, List[str]]):
        """Validate that containerized growth topology uses the same hostname/IP for all components (all-in-one)."""
        # Extract hostnames from each section (ignore host variables)
        hostnames = {}
        for section_name, hosts in sections.items():
            if section_name in ['automationgateway', 'automationcontroller', 'automationhub', 'automationeda',
                                'database']:
                if hosts:
                    # Extract just the hostname part (before any variables)
                    host = hosts[0].split()[0]
                    hostnames[section_name] = host

        # Check if all hostnames are the same
        if len(hostnames) > 1:
            unique_hostnames = set(hostnames.values())
            if len(unique_hostnames) > 1:
                self.errors.append(
                    f"Containerized growth topology should use the same hostname/IP for all components (all-in-one). Found different hostnames: {dict(hostnames)}")


class InventoryComparator(InventoryProcessor):
    """Compares two AAP inventory files for semantic equivalence."""

    def compare_inventories(self, inventory1_path: str, inventory2_path: str) -> bool:
        """Compare two inventory files for semantic equivalence."""
        try:
            sections1, variables1 = self.parse_inventory(inventory1_path)
            sections2, variables2 = self.parse_inventory(inventory2_path)
        except Exception as e:
            self.errors.append(str(e))
            return False

        # Compare sections
        sections_equal = self._compare_sections(sections1, sections2)

        # Compare variables
        variables_equal = self._compare_variables(variables1, variables2)

        return sections_equal and variables_equal and len(self.errors) == 0

    def _compare_sections(self, sections1: Dict[str, List[str]], sections2: Dict[str, List[str]]) -> bool:
        """Compare sections between two inventories."""
        all_sections = set(sections1.keys()) | set(sections2.keys())
        sections_equal = True

        for section in all_sections:
            if section not in sections1:
                self.errors.append(f"Section [{section}] missing in first inventory")
                sections_equal = False
            elif section not in sections2:
                self.errors.append(f"Section [{section}] missing in second inventory")
                sections_equal = False
            else:
                # Normalize hosts by sorting (order doesn't matter for semantic equivalence)
                hosts1 = sorted(sections1[section])
                hosts2 = sorted(sections2[section])

                if hosts1 != hosts2:
                    self.errors.append(f"Section [{section}] differs between inventories")
                    self.errors.append(f"  First inventory: {hosts1}")
                    self.errors.append(f"  Second inventory: {hosts2}")
                    sections_equal = False

        return sections_equal

    def _compare_variables(self, variables1: Dict[str, str], variables2: Dict[str, str]) -> bool:
        """Compare variables between two inventories."""
        all_vars = set(variables1.keys()) | set(variables2.keys())
        variables_equal = True

        for var in all_vars:
            if var not in variables1:
                self.errors.append(f"Variable '{var}' missing in first inventory")
                variables_equal = False
            elif var not in variables2:
                self.errors.append(f"Variable '{var}' missing in second inventory")
                variables_equal = False
            else:
                val1 = variables1[var].strip()
                val2 = variables2[var].strip()

                if val1 != val2:
                    self.errors.append(f"Variable '{var}' differs between inventories")
                    self.errors.append(f"  First inventory: '{val1}'")
                    self.errors.append(f"  Second inventory: '{val2}'")
                    variables_equal = False

        return variables_equal


def validate_command(args):
    """Handle the validate subcommand."""
    if not Path(args.inventory).exists():
        print(f"Error: Inventory file not found: {args.inventory}")
        sys.exit(1)

    # Create validator and run validation
    validator = InventoryValidator(args.platform, args.topology)
    is_valid = validator.validate_inventory(args.inventory)

    # Get results
    results = validator.get_results()

    # Print results
    print(f"Validating {args.inventory} for {args.platform} {args.topology} topology...")
    print()

    if results['errors']:
        print("ERRORS:")
        for error in results['errors']:
            print(f"  {error}")
        print()

    if results['warnings']:
        print("WARNINGS:")
        for warning in results['warnings']:
            print(f"  {warning}")
        print()

    if is_valid:
        print("Inventory validation passed!")
        if results['warnings']:
            print("   (with warnings)")
        sys.exit(0)
    else:
        print("Inventory validation failed!")
        sys.exit(1)


def compare_command(args):
    """Handle the compare subcommand."""
    if not Path(args.inventory1).exists():
        print(f"Error: First inventory file not found: {args.inventory1}")
        sys.exit(1)

    if not Path(args.inventory2).exists():
        print(f"Error: Second inventory file not found: {args.inventory2}")
        sys.exit(1)

    # Create comparator and run comparison
    comparator = InventoryComparator()
    are_equivalent = comparator.compare_inventories(args.inventory1, args.inventory2)

    # Get results
    results = comparator.get_results()

    # Print results
    print(f"Comparing {args.inventory1} and {args.inventory2}...")
    print()

    if results['errors']:
        print("DIFFERENCES:")
        for error in results['errors']:
            print(f"  {error}")
        print()

    if results['warnings']:
        print("WARNINGS:")
        for warning in results['warnings']:
            print(f"  {warning}")
        print()

    if are_equivalent:
        print("Inventories are semantically equivalent!")
        sys.exit(0)
    else:
        print("Inventories are not semantically equivalent!")
        sys.exit(1)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description='AAP Inventory Tool - Validate and compare AAP inventory files'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Validate subcommand
    validate_parser = subparsers.add_parser('validate', help='Validate an inventory file')
    validate_parser.add_argument(
        '--inventory',
        required=True,
        help='Path to the inventory file to validate'
    )
    validate_parser.add_argument(
        '--platform',
        required=True,
        choices=['containerized', 'rpm'],
        help='Platform type (containerized or rpm)'
    )
    validate_parser.add_argument(
        '--topology',
        required=True,
        choices=['growth', 'enterprise'],
        help='Topology type (growth or enterprise)'
    )

    # Compare subcommand
    compare_parser = subparsers.add_parser('compare', help='Compare two inventory files')
    compare_parser.add_argument(
        '--inventory1',
        required=True,
        help='Path to the first inventory file'
    )
    compare_parser.add_argument(
        '--inventory2',
        required=True,
        help='Path to the second inventory file'
    )

    # Parse arguments
    args = parser.parse_args()

    if args.command == 'validate':
        validate_command(args)
    elif args.command == 'compare':
        compare_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()