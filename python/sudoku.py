import re
import json
from typing import Dict, List, Optional
from pydantic import BaseModel, ValidationError

# ----------------------------
# Pydantic Model for a Sudoku Cell
# ----------------------------
class SudokuCell(BaseModel):
    value: Optional[int] = None  # Solved digit (1–9) or None if unsolved.
    candidates: List[int] = []   # List of candidate digits.

# Type alias for our puzzle (a mapping from cell key to cell record)
SudokuPuzzle = Dict[str, dict]


# ----------------------------
# 1. Reading and Building the Puzzle
# ----------------------------
def read_puzzle_from_file(file_path: str) -> SudokuPuzzle:
    """
    Reads a Sudoku puzzle from a text file and returns a JSON-compatible dict.
    Each key is in the form "R{row}C{col}" (e.g. "R1C1"). In the file, solved cells
    contain a digit and unsolved cells are marked with '_' or '0'.
    """
    puzzle: Dict[str, dict] = {}
    rows = []
    
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines or separator lines (e.g. "---------------------")
            if not line or line.startswith("-"):
                continue
            # Remove vertical bar characters and extra spaces.
            line = line.replace("|", "")
            tokens = line.split()
            if tokens:
                rows.append(tokens)
    
    if len(rows) != 9:
        raise ValueError(f"Expected 9 rows in the puzzle, got {len(rows)}")
    
    # Process each row.
    for row_idx, tokens in enumerate(rows, start=1):
        if len(tokens) != 9:
            raise ValueError(f"Expected 9 tokens in row {row_idx}, got {len(tokens)}")
        for col_idx, token in enumerate(tokens, start=1):
            cell_key = f"R{row_idx}C{col_idx}"
            if token in ("_", "0"):
                cell_value = None
            else:
                try:
                    cell_value = int(token)
                except ValueError:
                    raise ValueError(f"Invalid token '{token}' in cell {cell_key}.")
            # Create the cell record.
            cell = SudokuCell(value=cell_value, candidates=[])
            puzzle[cell_key] = cell.dict()
    return puzzle


def get_units_for_cell(cell_ref: str) -> Dict[str, List[str]]:
    """
    Given a cell reference (e.g. "R4C1"), return a dict containing the keys:
      - 'row': all cell keys in the same row,
      - 'col': all cell keys in the same column,
      - 'block': all cell keys in the same 3x3 block.
    (The cell itself appears in each list.)
    """
    m = re.match(r"R(\d+)C(\d+)", cell_ref)
    if not m:
        raise ValueError(f"Invalid cell reference: {cell_ref}")
    row_num = int(m.group(1))
    col_num = int(m.group(2))
    
    row_unit = [f"R{row_num}C{c}" for c in range(1, 10)]
    col_unit = [f"R{r}C{col_num}" for r in range(1, 10)]
    
    # Calculate top-left coordinate for the block.
    block_row = (row_num - 1) // 3
    block_col = (col_num - 1) // 3
    row_start = block_row * 3 + 1
    col_start = block_col * 3 + 1
    block_unit = [
        f"R{r}C{c}"
        for r in range(row_start, row_start + 3)
        for c in range(col_start, col_start + 3)
    ]
    
    return {"row": row_unit, "col": col_unit, "block": block_unit}


# ----------------------------
# 2. Computing Candidate Lists and Assignments
# ----------------------------
def compute_candidates(puzzle: SudokuPuzzle) -> SudokuPuzzle:
    """
    For each unsolved cell in the puzzle, compute its candidate digits 
    (digits 1-9 that do not already appear in its row, column, or block).
    Updates the puzzle in place.
    """
    for cell_ref, cell in puzzle.items():
        if cell["value"] is not None:
            cell["candidates"] = []
            continue
        used_digits = set()
        units = get_units_for_cell(cell_ref)
        for unit in units.values():
            for peer in unit:
                peer_value = puzzle[peer]["value"]
                if peer_value is not None:
                    used_digits.add(peer_value)
        cell["candidates"] = [d for d in range(1, 10) if d not in used_digits]
    return puzzle


def assign_digit(puzzle: SudokuPuzzle, cell_ref: str, digit: int) -> SudokuPuzzle:
    """
    Assigns a digit to a cell in the puzzle. It first checks that the cell is unsolved
    and that the digit is in its candidate list. Then, it sets the cell's value and removes
    that digit from the candidate lists of all unsolved peers (in row, column, and block).
    If any peer is reduced to a single candidate, that digit is automatically assigned.
    """
    cell = puzzle.get(cell_ref)
    if cell is None:
        raise ValueError(f"Cell {cell_ref} not found in the puzzle.")
    if cell["value"] is not None:
        raise ValueError(f"Cell {cell_ref} is already assigned with value {cell['value']}.")
    if digit not in cell["candidates"]:
        raise ValueError(f"Digit {digit} is not a candidate for cell {cell_ref}.")

    # Assign the digit.
    cell["value"] = digit
    cell["candidates"] = []

    # Eliminate the digit from all peers.
    units = get_units_for_cell(cell_ref)
    for unit in units.values():
        for peer_ref in unit:
            if peer_ref == cell_ref:
                continue
            peer = puzzle[peer_ref]
            if peer["value"] is None and digit in peer["candidates"]:
                peer["candidates"].remove(digit)
                # Automatically assign if only one candidate remains.
                if len(peer["candidates"]) == 1:
                    sole_candidate = peer["candidates"][0]
                    assign_digit(puzzle, peer_ref, sole_candidate)
    return puzzle


def eliminate_digit(puzzle: SudokuPuzzle, cell_ref: str, digit: int) -> SudokuPuzzle:
    """
    Eliminates a digit from a cell’s candidate list (if the cell is unsolved).
    """
    cell = puzzle.get(cell_ref)
    if cell is None:
        raise ValueError(f"Cell {cell_ref} not found in the puzzle.")
    if cell["value"] is not None:
        raise ValueError(f"Cannot eliminate digit from cell {cell_ref} because it is already solved.")
    if digit in cell["candidates"]:
        cell["candidates"].remove(digit)
    return puzzle


def scan_and_assign(puzzle: SudokuPuzzle) -> SudokuPuzzle:
    """
    Scans the puzzle repeatedly and, if an unsolved cell has exactly one candidate,
    assigns that candidate. Continues until no further single-candidate cells are found.
    """
    progress = True
    while progress:
        progress = False
        # Copy keys to avoid runtime modification issues.
        for cell_ref, cell in list(puzzle.items()):
            if cell["value"] is None and len(cell["candidates"]) == 1:
                candidate = cell["candidates"][0]
                assign_digit(puzzle, cell_ref, candidate)
                progress = True
    return puzzle


# ----------------------------
# 3. Additional Utility Functions
# ----------------------------
def get_unit(puzzle: SudokuPuzzle, unit_ref: str) -> Dict[str, dict]:
    """
    Given a unit reference as a string ("R1", "C1", or "B1"), returns a dictionary
    of all cell keys and their corresponding cell records in that unit.
    
    - For rows: "R1" returns row 1 (cells R1C1, R1C2, …, R1C9).
    - For columns: "C1" returns column 1 (cells R1C1, R2C1, …, R9C1).
    - For blocks: "B1" returns the top-left block (cells R1C1–R3C3), "B2" returns the
      top-middle block, and so on.
    """
    unit_type = unit_ref[0].upper()
    try:
        index = int(unit_ref[1:])
    except ValueError:
        raise ValueError(f"Invalid unit reference: {unit_ref}")
    
    if unit_type == "R":
        keys = [f"R{index}C{c}" for c in range(1, 10)]
    elif unit_type == "C":
        keys = [f"R{r}C{index}" for r in range(1, 10)]
    elif unit_type == "B":
        # Calculate block starting positions.
        # Blocks are numbered 1-9 left-to-right, top-to-bottom.
        row_start = ((index - 1) // 3) * 3 + 1
        col_start = ((index - 1) % 3) * 3 + 1
        keys = [f"R{r}C{c}" for r in range(row_start, row_start + 3)
                              for c in range(col_start, col_start + 3)]
    else:
        raise ValueError("Unit reference must start with 'R', 'C', or 'B'.")
    
    return {k: puzzle[k] for k in keys if k in puzzle}


def render_puzzle(puzzle: SudokuPuzzle, as_markdown: bool = True, show_candidates: bool = False) -> str:
    """
    Renders the Sudoku puzzle as a table.
    
    - When show_candidates is False, solved cells display their value and unsolved cells show '_'.
    - When show_candidates is True, unsolved cells display their candidate digits in a bracketed, comma-separated format,
      e.g. {1, 2, 3}.
      
    If as_markdown is True, the output is formatted as a Markdown table with column headers and row labels.
    Otherwise, a plain text rendering is produced.
    """
    if as_markdown:
        # Build a Markdown table with a header row.
        header = [""] + [f"C{col}" for col in range(1, 10)]
        table_rows = [header]
        # Markdown table header separator.
        table_rows.append(["---"] * len(header))
        # Build each row with a row label in the first column.
        for r in range(1, 10):
            row_label = f"R{r}"
            row_data = [row_label]
            for c in range(1, 10):
                cell = puzzle[f"R{r}C{c}"]
                if cell["value"] is not None:
                    cell_str = str(cell["value"])
                else:
                    if show_candidates and cell["candidates"]:
                        # Sort candidates and display them in a bracketed, comma-separated format.
                        candidates = sorted(cell["candidates"])
                        candidate_str = ", ".join(str(d) for d in candidates)
                        cell_str = "{" + candidate_str + "}"
                    else:
                        cell_str = "_"
                row_data.append(cell_str)
            table_rows.append(row_data)
        # Convert rows to Markdown table format.
        lines = []
        for row in table_rows:
            line = "| " + " | ".join(row) + " |"
            lines.append(line)
        result = "\n".join(lines)
        return result
    else:
        # Plain text rendering with block separators.
        lines = []
        for r in range(1, 10):
            row_cells = []
            for c in range(1, 10):
                cell = puzzle[f"R{r}C{c}"]
                if cell["value"] is not None:
                    ch = str(cell["value"])
                else:
                    if show_candidates and cell["candidates"]:
                        candidates = sorted(cell["candidates"])
                        candidate_str = ", ".join(str(d) for d in candidates)
                        ch = "{" + candidate_str + "}"
                    else:
                        ch = "_"
                row_cells.append(ch)
            # Insert vertical separators between blocks.
            row_str = " | ".join([" ".join(row_cells[i:i+3]) for i in range(0, 9, 3)])
            lines.append(row_str)
            if r % 3 == 0 and r < 9:
                lines.append("-" * len(row_str))
        return "\n".join(lines)


def check_strict_consistency(puzzle: SudokuPuzzle) -> bool:
    """
    Checks strict consistency: in every unit (row, column, and block), no solved digit
    appears more than once.
    
    Returns True if the puzzle is strictly consistent; otherwise, prints an error and returns False.
    """
    # Create a list of all unit references.
    unit_refs = [f"R{i}" for i in range(1, 10)] + \
                [f"C{i}" for i in range(1, 10)] + \
                [f"B{i}" for i in range(1, 10)]
    for unit_ref in unit_refs:
        unit_cells = get_unit(puzzle, unit_ref)
        seen = {}
        for cell_key, cell in unit_cells.items():
            if cell["value"] is not None:
                digit = cell["value"]
                if digit in seen:
                    print(f"Strict consistency error in {unit_ref}: digit {digit} appears in both {seen[digit]} and {cell_key}.")
                    return False
                seen[digit] = cell_key
    return True


def check_candidate_consistency(puzzle: SudokuPuzzle) -> bool:
    """
    Checks candidate consistency: in every unit (row, column, block), for every digit 1–9,
    either the digit is already solved in that unit, or it appears in at least one unsolved cell's
    candidate list.
    
    Returns True if every unit passes this check; otherwise, prints an error and returns False.
    """
    unit_refs = [f"R{i}" for i in range(1, 10)] + \
                [f"C{i}" for i in range(1, 10)] + \
                [f"B{i}" for i in range(1, 10)]
    for unit_ref in unit_refs:
        unit_cells = get_unit(puzzle, unit_ref)
        solved_digits = {cell["value"] for cell in unit_cells.values() if cell["value"] is not None}
        for d in range(1, 10):
            # If the digit is not already solved in the unit, check unsolved cells.
            if d not in solved_digits:
                digit_found = any(d in cell["candidates"] for cell in unit_cells.values() if cell["value"] is None)
                if not digit_found:
                    print(f"Candidate consistency error in {unit_ref}: digit {d} is missing (neither solved nor penciled in any unsolved cell).")
                    return False
    return True


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == "__main__":
    file_path = "sudoku.txt"  # Ensure this file exists and is formatted properly.
    try:
        # Step 1: Read the puzzle from a text file.
        puzzle = read_puzzle_from_file(file_path)
        print("Initial Puzzle (JSON):")
        print(json.dumps(puzzle, indent=4))
    
        # Step 2: Compute candidate lists for unsolved cells.
        puzzle = compute_candidates(puzzle)
        print("\nPuzzle with Candidate Lists:")
        print(json.dumps(puzzle, indent=4))
    
        # Demonstrate retrieving a specific unit.
        print("\nExtracting Row 1:")
        row1 = get_unit(puzzle, "R1")
        print(json.dumps(row1, indent=4))
    
        print("\nExtracting Column 5:")
        col5 = get_unit(puzzle, "C5")
        print(json.dumps(col5, indent=4))
    
        print("\nExtracting Block 1:")
        block1 = get_unit(puzzle, "B1")
        print(json.dumps(block1, indent=4))
    
        # Step 3: Render the puzzle as a Markdown table.
        # Toggle show_candidates=True to display candidate lists in unsolved cells.
        rendered_markdown = render_puzzle(puzzle, as_markdown=True, show_candidates=True)
        print("\nRendered Puzzle as Markdown Table (with candidate lists):")
        print(rendered_markdown)
    
        # Step 4: Check strict consistency.
        if check_strict_consistency(puzzle):
            print("\nStrict consistency check passed.")
        else:
            print("\nStrict consistency check failed.")
    
        # Step 5: Check candidate consistency.
        if check_candidate_consistency(puzzle):
            print("Candidate consistency check passed.")
        else:
            print("Candidate consistency check failed.")
    
        # Optionally, scan the board and auto-assign single-candidate cells.
        puzzle = scan_and_assign(puzzle)
        print("\nPuzzle after scanning and auto-assigning single-candidate cells:")
        print(render_puzzle(puzzle, as_markdown=True, show_candidates=True))
    
    except (ValueError, ValidationError) as e:
        print("Error:", e)