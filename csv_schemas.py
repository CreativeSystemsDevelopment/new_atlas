"""Central schema definitions for selected CSV inputs in the project.

This module contains:
- 3 inventory schemas: die_inventory.csv, maintenance_inventory.csv, machine_inventory.csv
- 3 downtime schemas: indirect, die, and machine downtime logs

Use this as a single source of truth for validation, ETL, and docs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class FieldSpec:
    """Schema for one CSV column."""
    name: str
    dtype: str
    required: bool = True
    description: str = ""


@dataclass(frozen=True)
class CsvSchema:
    """Schema for one CSV file."""
    filename: str
    purpose: str
    description: str
    fields: List[FieldSpec]
    notes: Optional[str] = None

    @property
    def required_fields(self) -> List[str]:
        return [f.name for f in self.fields if f.required]


INVENTORY_SCHEMAS: Dict[str, CsvSchema] = {
    "die_inventory.csv": CsvSchema(
        filename="die_inventory.csv",
        purpose="inventory.parts.die",
        description=(
            "Electrical part-level list for the die machine project. "
            "Useful for linking symbol text/part numbers to panel BOMs."
        ),
        fields=[
            FieldSpec("Location", "string", True, "Top-level physical area or panel area"),
            FieldSpec("Symbol Text", "string", True, "Symbol/marker label used in schematics"),
            FieldSpec("Description", "string", True, "Text description of component"),
            FieldSpec("Part Number", "string", True, "Vendor/manufacturer part number"),
            FieldSpec("Quantity", "integer", True, "Quantity in the part line"),
            FieldSpec("Manufacturer", "string", True, "Vendor/manufacturer"),
            FieldSpec("Notes", "string", True, "Notes/comments"),
            FieldSpec("Number", "integer", True, "Line sequence number"),
            FieldSpec("Drawing Number", "string", True, "Drawing reference"),
            FieldSpec("Equipment", "string", True, "Associated equipment name"),
            FieldSpec("Order Number", "integer", True, "Associated order identifier"),
        ],
    ),
    "maintenance_inventory.csv": CsvSchema(
        filename="maintenance_inventory.csv",
        purpose="inventory.parts.maintenance",
        description="Maintenance parts snapshot with cost and organization details.",
        fields=[
            FieldSpec("Item", "string", True, "Unique maintenance item code"),
            FieldSpec("Item Description", "string", True, "Part description"),
            FieldSpec(
                "Item Status",
                "string",
                True,
                "Active/Obsolete/Inacive/NoPurchase/etc.",
            ),
            FieldSpec("Quantity", "integer", True, "On-hand quantity"),
            FieldSpec(
                "Subinventory",
                "string",
                True,
                "Subinventory/location bucket inside the stock system",
            ),
            FieldSpec("UOM", "string", True, "Unit of measure"),
            FieldSpec("Cost", "decimal", True, "Unit cost when present"),
            FieldSpec("Organization", "string", True, "Owning org/location"),
        ],
    ),
    "machine_inventory.csv": CsvSchema(
        filename="machine_inventory.csv",
        purpose="inventory.parts.machine",
        description="Machine-spare inventory across physical AAC plants.",
        fields=[
            FieldSpec("Item", "string", True, "Unique machine part code"),
            FieldSpec("Item Description", "string", True, "Part description"),
            FieldSpec(
                "Item Status",
                "string",
                True,
                "Active/Inactive/NoPurchase/Obsolete/etc.",
            ),
            FieldSpec("Quantity", "integer", True, "On-hand quantity"),
            FieldSpec("Subinventory", "string", True, "Subinventory bin"),
            FieldSpec("UOM", "string", True, "Unit of measure"),
            FieldSpec("Location", "string", True, "Shelf/bin location"),
            FieldSpec("Lot Number", "string", True, "Lot identifier if available"),
            FieldSpec("Organization", "string", True, "Owning org/location"),
        ],
    ),
}


DOWNTIME_SCHEMAS: Dict[str, CsvSchema] = {
    "mar2022_jun2024_indirect_downtime.csv": CsvSchema(
        filename="mar2022_jun2024_indirect_downtime.csv",
        purpose="downtime.indirect",
        description="Indirect downtime events (non-machine and non-die specific categories).",
        fields=[
            FieldSpec("Date", "date", True, "Event date in d-MMM-yy (example: 1-Mar-22)"),
            FieldSpec("Plant", "string", True, "Plant site"),
            FieldSpec("Shift", "integer", True, "Shift code"),
            FieldSpec("Machine Number", "string", True, "Machine identifier"),
            FieldSpec("Part Description", "string", True, "Part being produced"),
            FieldSpec("Die Number", "integer", True, "Die identifier"),
            FieldSpec(
                "Indirect Downtime Reason",
                "string",
                True,
                "Categorized reason (e.g., Clean Up, Die Change, Other).",
            ),
            FieldSpec(
                "Indirect Downtime Minutes",
                "integer",
                True,
                "Duration in minutes",
            ),
            FieldSpec("Indirect Downtime Notes", "string", True, "Free text notes"),
            FieldSpec("ID DT Description", "string", True, "Reason grouping/category"),
            FieldSpec(
                "ID Start Time",
                "time",
                True,
                "Clock time at downtime start (hh:mm AM/PM)",
            ),
            FieldSpec(
                "ID Stop Time",
                "time",
                True,
                "Clock time at downtime stop (hh:mm AM/PM)",
            ),
        ],
    ),
    "mar2022_jun2024_die_downtime.csv": CsvSchema(
        filename="mar2022_jun2024_die_downtime.csv",
        purpose="downtime.die",
        description="Die-related downtime events.",
        fields=[
            FieldSpec("Date", "date", True, "Event date in d-MMM-yy (example: 1-Mar-22)"),
            FieldSpec("Plant", "string", True, "Plant site"),
            FieldSpec("Shift", "integer", True, "Shift code"),
            FieldSpec("Machine Number", "string", True, "Machine identifier"),
            FieldSpec("Part Description", "string", True, "Part being produced"),
            FieldSpec("Die Number", "integer", True, "Die identifier"),
            FieldSpec(
                "Die Related Downtime Reason",
                "string",
                True,
                "Die-specific reason category",
            ),
            FieldSpec(
                "Die Related Downtime Minutes",
                "integer",
                True,
                "Duration in minutes",
            ),
            FieldSpec("Die Related Downtime Notes", "string", True, "Free text notes"),
            FieldSpec("DRD DT Description", "string", True, "Reason grouping/category"),
            FieldSpec(
                "DRD Start Time",
                "time",
                True,
                "Clock time at downtime start (hh:mm AM/PM)",
            ),
            FieldSpec(
                "DRD Stop Time",
                "time",
                True,
                "Clock time at downtime stop (hh:mm AM/PM)",
            ),
        ],
    ),
    "mar2022_jun2024_machine_downtime.csv": CsvSchema(
        filename="mar2022_jun2024_machine_downtime.csv",
        purpose="downtime.machine",
        description="Machine-related downtime events.",
        fields=[
            FieldSpec("Date", "date", True, "Event date in d-MMM-yy (example: 1-Mar-22)"),
            FieldSpec("Plant", "string", True, "Plant site"),
            FieldSpec("Shift", "integer", True, "Shift code"),
            FieldSpec("Machine Number", "string", True, "Machine identifier"),
            FieldSpec("Part Description", "string", True, "Part being produced"),
            FieldSpec("Die Number", "integer", True, "Die identifier"),
            FieldSpec(
                "Machine Related Downtime Reason",
                "string",
                True,
                "Machine/equipment reason category",
            ),
            FieldSpec(
                "Machine Related Downtime Minutes",
                "integer",
                True,
                "Duration in minutes",
            ),
            FieldSpec("Machine Related Downtime Notes", "string", True, "Free text notes"),
            FieldSpec("MRD DT Description", "string", True, "Reason grouping/category"),
            FieldSpec(
                "MRD Start Time",
                "time",
                True,
                "Clock time at downtime start (hh:mm AM/PM)",
            ),
            FieldSpec(
                "MRD Stop Time",
                "time",
                True,
                "Clock time at downtime stop (hh:mm AM/PM)",
            ),
        ],
    ),
}


ALL_CSV_SCHEMAS: Dict[str, CsvSchema] = {
    **{f"inventory/{k}": v for k, v in INVENTORY_SCHEMAS.items()},
    **{f"downtime/{k}": v for k, v in DOWNTIME_SCHEMAS.items()},
}


def get_schema(filename: str) -> Optional[CsvSchema]:
    """Return a schema by filename."""
    return (
        INVENTORY_SCHEMAS.get(filename)
        or DOWNTIME_SCHEMAS.get(filename)
        or ALL_CSV_SCHEMAS.get(filename)
    )


def list_inventory_csvs() -> List[CsvSchema]:
    """Return all inventory CSV schemas."""
    return list(INVENTORY_SCHEMAS.values())


def list_downtime_csvs() -> List[CsvSchema]:
    """Return all downtime CSV schemas."""
    return list(DOWNTIME_SCHEMAS.values())

