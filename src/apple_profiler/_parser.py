"""XML parser with id/ref resolution for xctrace export output.

xctrace export XML uses a deduplication scheme:
- First occurrence: `<thread id="2" fmt="Main Thread">` with child elements
- Subsequent:       `<thread ref="2"/>` with no children/text

This module resolves all refs to their original elements, producing a unified
tree where every element has its full data regardless of whether it was the
first or subsequent occurrence.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass, field


def _empty_str_dict() -> dict[str, str]:
    return {}


@dataclass
class ResolvedElement:
    """A fully-resolved XML element with all id/ref lookups completed."""

    tag: str
    text: str | None = None
    fmt: str | None = None
    attrs: dict[str, str] = field(default_factory=_empty_str_dict)
    children: list[ResolvedElement] = field(default_factory=lambda: [])

    def child(self, tag: str) -> ResolvedElement | None:
        """Find first child with given tag."""
        for c in self.children:
            if c.tag == tag:
                return c
        return None

    def children_by_tag(self, tag: str) -> list[ResolvedElement]:
        """Find all children with given tag."""
        return [c for c in self.children if c.tag == tag]

    @property
    def value(self) -> str:
        """The display value: fmt if available, else text, else empty string."""
        if self.fmt is not None:
            return self.fmt
        if self.text is not None:
            return self.text
        return ""

    @property
    def int_value(self) -> int:
        """Parse text as integer, defaulting to 0."""
        if self.text is not None:
            try:
                return int(self.text)
            except ValueError:
                pass
        return 0

    def __repr__(self) -> str:
        parts = [f"<{self.tag}"]
        if self.fmt:
            parts.append(f' fmt="{self.fmt}"')
        if self.text:
            parts.append(f">{self.text}</{self.tag}>")
        elif self.children:
            parts.append(f"> ({len(self.children)} children)")
        else:
            parts.append("/>")
        return "".join(parts)


@dataclass
class SchemaColumn:
    """A column definition from the schema header."""

    mnemonic: str
    name: str
    engineering_type: str


@dataclass
class ParsedTable:
    """A parsed xctrace data table with schema and resolved rows."""

    schema_name: str
    columns: list[SchemaColumn]
    rows: list[list[ResolvedElement]]


def _resolve_element(elem: ET.Element, lookup: dict[str, ET.Element]) -> ResolvedElement:
    """Resolve a single XML element, following refs as needed."""
    ref = elem.get("ref")
    if ref is not None:
        source = lookup.get(ref)
        if source is None:
            return ResolvedElement(tag=elem.tag, attrs=dict(elem.attrib))
        return _resolve_element(source, lookup)

    attrs = dict(elem.attrib)
    resolved = ResolvedElement(
        tag=elem.tag,
        text=elem.text.strip() if elem.text and elem.text.strip() else None,
        fmt=attrs.pop("fmt", None),
        attrs=attrs,
    )

    # Remove 'id' from attrs since it's internal bookkeeping
    resolved.attrs.pop("id", None)

    for child in elem:
        if child.tag == "sentinel":
            continue
        resolved.children.append(_resolve_element(child, lookup))

    return resolved


def _build_id_lookup(root: ET.Element) -> dict[str, ET.Element]:
    """Build a lookup table of id -> element for all elements with id attrs."""
    lookup: dict[str, ET.Element] = {}
    for elem in root.iter():
        eid = elem.get("id")
        if eid is not None:
            lookup[eid] = elem
    return lookup


def parse_table_xml(xml_string: str) -> ParsedTable:
    """Parse an xctrace table export XML string into a ParsedTable.

    The XML is expected to be the output of `xctrace export --xpath` for a
    single table, wrapped in <trace-query-result><node>...</node></trace-query-result>.
    """
    root = ET.fromstring(xml_string)
    lookup = _build_id_lookup(root)

    # Find the schema element
    node = root.find(".//node")
    if node is None:
        node = root

    schema_elem = node.find("schema")
    schema_name = ""
    columns: list[SchemaColumn] = []

    if schema_elem is not None:
        schema_name = schema_elem.get("name", "")
        for col in schema_elem.findall("col"):
            mnemonic_el = col.find("mnemonic")
            name_el = col.find("name")
            eng_type_el = col.find("engineering-type")
            columns.append(
                SchemaColumn(
                    mnemonic=mnemonic_el.text
                    if mnemonic_el is not None and mnemonic_el.text
                    else "",
                    name=name_el.text if name_el is not None and name_el.text else "",
                    engineering_type=(
                        eng_type_el.text if eng_type_el is not None and eng_type_el.text else ""
                    ),
                )
            )

    rows: list[list[ResolvedElement]] = []
    for row_elem in node.findall("row"):
        row: list[ResolvedElement] = []
        for child in row_elem:
            if child.tag == "sentinel":
                row.append(ResolvedElement(tag="sentinel"))
                continue
            row.append(_resolve_element(child, lookup))
        rows.append(row)

    return ParsedTable(schema_name=schema_name, columns=columns, rows=rows)


def parse_toc_xml(xml_string: str) -> ET.Element:
    """Parse a TOC XML string and return the root element."""
    return ET.fromstring(xml_string)


def iter_rows(xml_string: str) -> Iterator[list[ResolvedElement]]:
    """Memory-efficient row iterator using iterparse.

    For very large tables, use this instead of parse_table_xml to avoid
    loading everything into memory at once. Note: id/ref resolution still
    requires building the full lookup, so this only saves on the resolved
    row storage.
    """
    root = ET.fromstring(xml_string)
    lookup = _build_id_lookup(root)

    node = root.find(".//node")
    if node is None:
        node = root

    for row_elem in node.findall("row"):
        row: list[ResolvedElement] = []
        for child in row_elem:
            if child.tag == "sentinel":
                row.append(ResolvedElement(tag="sentinel"))
                continue
            row.append(_resolve_element(child, lookup))
        yield row
