from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from typing import Dict, List, Optional
import re
import ssl

app = FastAPI(
    title = "Sudoku Microservice",
    description = "API endpoints for interacting with a sudoku puzzle."
)
ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain('certs/geraldyong-cert.pem', keyfile='certs/geraldyong-priv.pem')

# Enable CORS
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],  # Adjust for production
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

# -----------------------------------------
# Pydantic Models for Sudoku and API Inputs
# -----------------------------------------
class SudokuCell(BaseModel):
    value: Optional[int] = None  # Solved digit (1-9) or None if unsolved.
    candidates: List[int] = []   # Candidate digits list.

# The puzzle is represented as a dict mapping cell keys (e.g. "R1C1") to SudokuCell
# For API endpoints we wrap the puzzle in a model.
class PuzzleInput(BaseModel):
    puzzle: Dict[str, SudokuCell]

class PuzzleTextInput(BaseModel):
    text: str

class CellAction(BaseModel):
    puzzle: Dict[str, SudokuCell]
    cell_ref: str
    digit: int

class UnitRequest(BaseModel):
    puzzle: Dict[str, SudokuCell]
    unit_ref: str  # e.g., "R1", "C5", or "B1"

class RenderRequest(BaseModel):
    puzzle: Dict[str, SudokuCell]
    as_markdown: bool = True
    show_candidates: bool = False

class CheckResult(BaseModel):
    result: bool
    message: Optional[str] = None

# -------------------------------------------------
# Helper: Convert API-sent puzzle to internal format
# -------------------------------------------------
def convert_puzzle(puzzle_in: Dict[str, SudokuCell]) -> Dict[str, dict]:
    """Convert each SudokuCell model to a dictionary."""
    new_puzzle = {}
    for k, v in puzzle_in.items():
        # If already a dict, leave it; otherwise convert via .dict()
        new_puzzle[k] = v if isinstance(v, dict) else v.dict()
    return new_puzzle

# -----------------------------------------
# Core Sudoku Functions (Logic)
# -----------------------------------------
def read_puzzle_from_text(text: str) -> Dict[str, dict]:
    """
    Parses a Sudoku puzzle from a multiline text string.
    Each row should have 9 tokens (digits or '_' or '0') with optional
    separators (like '|' or lines of dashes).
    """
    puzzle: Dict[str, dict] = {}
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("-"):
            continue
        line = line.replace("|", "")
        tokens = line.split()
        if tokens:
            rows.append(tokens)
    if len(rows) != 9:
        raise ValueError(f"Expected 9 rows in puzzle, got {len(rows)}")
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
    Given a cell reference (e.g. "R4C1"), returns a dict with keys:
      - 'row': all cell keys in the same row,
      - 'col': all cell keys in the same column,
      - 'block': all cell keys in the same 3×3 block.
    """
    m = re.match(r"R(\d+)C(\d+)", cell_ref)
    if not m:
        raise ValueError(f"Invalid cell reference: {cell_ref}")
    row_num = int(m.group(1))
    col_num = int(m.group(2))
    row_unit = [f"R{row_num}C{c}" for c in range(1, 10)]
    col_unit = [f"R{r}C{col_num}" for r in range(1, 10)]
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

def compute_candidates(puzzle: Dict[str, dict]) -> Dict[str, dict]:
    """
    For each unsolved cell, compute candidate digits (those not already in its row,
    column, or block) and update the puzzle in place.
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

def assign_digit(puzzle: Dict[str, dict], cell_ref: str, digit: int) -> Dict[str, dict]:
    """
    Assigns a digit to a cell if it is unsolved and the digit is in its candidate list.
    Also eliminates that digit from the candidate lists of all peers (row, col, block)
    and auto-assigns any peers reduced to a single candidate.
    """
    cell = puzzle.get(cell_ref)
    if cell is None:
        raise ValueError(f"Cell {cell_ref} not found in the puzzle.")
    if cell["value"] is not None:
        raise ValueError(f"Cell {cell_ref} is already solved with value {cell['value']}.")
    if digit not in cell["candidates"]:
        raise ValueError(f"Digit {digit} is not a candidate for cell {cell_ref}.")
    cell["value"] = digit
    cell["candidates"] = []
    units = get_units_for_cell(cell_ref)
    for unit in units.values():
        for peer_ref in unit:
            if peer_ref == cell_ref:
                continue
            peer = puzzle[peer_ref]
            if peer["value"] is None and digit in peer["candidates"]:
                peer["candidates"].remove(digit)
                if len(peer["candidates"]) == 1:
                    sole_candidate = peer["candidates"][0]
                    assign_digit(puzzle, peer_ref, sole_candidate)
    return puzzle

def eliminate_digit(puzzle: Dict[str, dict], cell_ref: str, digit: int) -> Dict[str, dict]:
    """
    Eliminates a digit from a cell’s candidate list (if the cell is unsolved).
    If the elimination leaves only one candidate, that candidate is automatically assigned.
    """
    cell = puzzle.get(cell_ref)
    if cell is None:
        raise ValueError(f"Cell {cell_ref} not found.")
    if cell["value"] is not None:
        raise ValueError(f"Cannot eliminate digit from cell {cell_ref} because it is already solved.")
    if digit in cell["candidates"]:
        cell["candidates"].remove(digit)
        # If only one candidate remains, assign that candidate automatically.
        if len(cell["candidates"]) == 1:
            sole_candidate = cell["candidates"][0]
            assign_digit(puzzle, cell_ref, sole_candidate)
    return puzzle

def scan_and_assign(puzzle: Dict[str, dict]) -> Dict[str, dict]:
    """
    Repeatedly scans the board and assigns a digit to any unsolved cell that has
    exactly one candidate.
    """
    progress = True
    while progress:
        progress = False
        for cell_ref, cell in list(puzzle.items()):
            if cell["value"] is None and len(cell["candidates"]) == 1:
                candidate = cell["candidates"][0]
                assign_digit(puzzle, cell_ref, candidate)
                progress = True
    return puzzle

def get_unit(puzzle: Dict[str, dict], unit_ref: str) -> Dict[str, dict]:
    """
    Returns a dictionary of cells for a given unit reference:
      - "R1" returns row 1,
      - "C1" returns column 1,
      - "B1" returns block 1 (top-left block).
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
        row_start = ((index - 1) // 3) * 3 + 1
        col_start = ((index - 1) % 3) * 3 + 1
        keys = [f"R{r}C{c}" for r in range(row_start, row_start + 3)
                              for c in range(col_start, col_start + 3)]
    else:
        raise ValueError("Unit reference must start with 'R', 'C', or 'B'.")
    return {k: puzzle[k] for k in keys if k in puzzle}

def render_puzzle(puzzle: Dict[str, dict], as_markdown: bool = True, show_candidates: bool = False) -> str:
    """
    Renders the puzzle as a table.
    
    For solved cells, the digit is shown. For unsolved cells:
      - If show_candidates is False, displays an underscore '_'.
      - If show_candidates is True, displays the sorted candidate list in the format: {1, 2, 3}.
    
    When as_markdown is True, the output is a Markdown table with column headers and row labels.
    """
    if as_markdown:
        header = [""] + [f"C{col}" for col in range(1, 10)]
        table_rows = [header, ["---"] * len(header)]
        for r in range(1, 10):
            row_label = f"R{r}"
            row_data = [row_label]
            for c in range(1, 10):
                cell = puzzle[f"R{r}C{c}"]
                if cell["value"] is not None:
                    cell_str = str(cell["value"])
                else:
                    if show_candidates and cell["candidates"]:
                        candidates = sorted(cell["candidates"])
                        candidate_str = ", ".join(str(d) for d in candidates)
                        cell_str = "{" + candidate_str + "}"
                    else:
                        cell_str = "_"
                row_data.append(cell_str)
            table_rows.append(row_data)
        lines = ["| " + " | ".join(row) + " |" for row in table_rows]
        return "\n".join(lines)
    else:
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
            row_str = " | ".join([" ".join(row_cells[i:i+3]) for i in range(0, 9, 3)])
            lines.append(row_str)
            if r % 3 == 0 and r < 9:
                lines.append("-" * len(row_str))
        return "\n".join(lines)

def check_strict_consistency(puzzle: Dict[str, dict]) -> bool:
    """
    In every unit (row, column, block), ensures no solved digit appears more than once.
    """
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
                    return False
                seen[digit] = cell_key
    return True

def check_candidate_consistency(puzzle: Dict[str, dict]) -> bool:
    """
    In every unit, for every digit 1-9, either the digit is solved in that unit
    or it appears in at least one unsolved cell's candidate list.
    """
    unit_refs = [f"R{i}" for i in range(1, 10)] + \
                [f"C{i}" for i in range(1, 10)] + \
                [f"B{i}" for i in range(1, 10)]
    for unit_ref in unit_refs:
        unit_cells = get_unit(puzzle, unit_ref)
        solved_digits = {cell["value"] for cell in unit_cells.values() if cell["value"] is not None}
        for d in range(1, 10):
            if d not in solved_digits:
                if not any(d in cell["candidates"] for cell in unit_cells.values() if cell["value"] is None):
                    return False
    return True

# -----------------------------------------
# FastAPI Endpoints
# -----------------------------------------
@app.post("/loadPuzzle", response_model=Dict[str, dict])
def load_puzzle_endpoint(input_data: PuzzleTextInput):
    """
    Upload a puzzle as text (with spaces, underscores, or zeros) and return the JSON representation.
    """
    try:
        puzzle = read_puzzle_from_text(input_data.text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return puzzle

@app.post("/computeCandidates", response_model=Dict[str, dict])
def compute_candidates_endpoint(input_data: PuzzleInput):
    puzzle_dict = convert_puzzle(input_data.puzzle)
    updated = compute_candidates(puzzle_dict)
    return updated

@app.post("/assignDigit", response_model=Dict[str, dict])
def assign_digit_endpoint(input_data: CellAction):
    puzzle_dict = convert_puzzle(input_data.puzzle)
    try:
        updated = assign_digit(puzzle_dict, input_data.cell_ref, input_data.digit)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return updated

@app.post("/eliminateDigit", response_model=Dict[str, dict])
def eliminate_digit_endpoint(input_data: CellAction):
    puzzle_dict = convert_puzzle(input_data.puzzle)
    try:
        updated = eliminate_digit(puzzle_dict, input_data.cell_ref, input_data.digit)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return updated

@app.post("/scanAndAssign", response_model=Dict[str, dict])
def scan_and_assign_endpoint(input_data: PuzzleInput):
    puzzle_dict = convert_puzzle(input_data.puzzle)
    updated = scan_and_assign(puzzle_dict)
    return updated

@app.post("/getUnit", response_model=Dict[str, dict])
def get_unit_endpoint(input_data: UnitRequest):
    puzzle_dict = convert_puzzle(input_data.puzzle)
    try:
        unit_data = get_unit(puzzle_dict, input_data.unit_ref)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return unit_data

@app.post("/renderPuzzle")
def render_puzzle_endpoint(input_data: RenderRequest):
    puzzle_dict = convert_puzzle(input_data.puzzle)
    rendered = render_puzzle(puzzle_dict, as_markdown=input_data.as_markdown, show_candidates=input_data.show_candidates)
    return {"rendered": rendered}

@app.post("/checkStrict", response_model=CheckResult)
def check_strict_endpoint(input_data: PuzzleInput):
    puzzle_dict = convert_puzzle(input_data.puzzle)
    result = check_strict_consistency(puzzle_dict)
    if result:
        return CheckResult(result=True, message="Strict consistency check passed.")
    else:
        return CheckResult(result=False, message="Strict consistency check failed.")

@app.post("/checkCandidates", response_model=CheckResult)
def check_candidates_endpoint(input_data: PuzzleInput):
    puzzle_dict = convert_puzzle(input_data.puzzle)
    result = check_candidate_consistency(puzzle_dict)
    if result:
        return CheckResult(result=True, message="Candidate consistency check passed.")
    else:
        return CheckResult(result=False, message="Candidate consistency check failed.")

# -----------------------------------------
# Run with Uvicorn when executed directly.
# -----------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

