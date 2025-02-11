from typing import Dict
from models import NextMove
from openai import OpenAI
from helper import *
import os
import json

# OpenAI API Key (ensure this is secured in a real-world app, like using environment variables)
client = OpenAI(
    api_key  = os.getenv('OPENAI_API_KEY'),
    organization = os.getenv('OPENAI_ORGANIZATION_ID')
)

# Strip pretext and posttext.
def strip_pre_post(text):
  # Find the start index of the first '[' character
  # and the end index of the first ']' character after '['
  start = text.find('[')
  end = text.rfind(']')

  # Check if both brackets were found
  if start != -1 and end != -1 and end > start:
    # Extract the text from '[' to ']'
    content = text[start + 1 : end]
  else:
    content = text

  return content


# -----------------------------------------
# LLM Agentic Function Schemas
# -----------------------------------------
get_cell_contents_schema = {
  "name": "get_cell_contents_schema_fn",
  "description": "If the cell is solved (its value is not null), returns the digit; otherwise, returns the list of candidate digits. Useful to determine the contents of a cell.",
  "parameters": {
    "type": "object",
    "properties": {
      "cell_ref": {
        "type": "string",
        "description": "The cell reference (e.g., 'R1C1') for which to retrieve the contents."
      }
    },
    "required": ["cell_ref"]
  }
}

find_assigned_peer_schema = {
    "name": "find_assigned_peer_schema_fn",
    "description": "Returns the peer cell that the specified digit is already assigned to, if available. Can be used to determine if the digit can be removed from this cell.",
    "parameters": {
        "type": "object",
        "properties": {
            "cell_ref": {
                "type": "string",
                "description": "The cell reference (e.g. 'R1C1')."
            },
            "digit": {
                "type": "integer",
                "description": "The digit to search for."
            }
        },
        "required": ["cell_ref", "digit"]
    }
}

find_candidate_peers_schema = {
    "name": "find_candidate_peers_schema_fn",
    "description": "Returns all the peer cells where the specified digit exists as one of the digits in their candidate list. Can be used to check if this cell is the only one among its peers to contain the digit in its candidate list.",
    "parameters": {
        "type": "object",
        "properties": {
            "cell_ref": {
                "type": "string",
                "description": "The cell reference (e.g. 'R1C1')."
            },
            "digit": {
                "type": "integer",
                "description": "The digit to search for in candidate lists."
            }
        },
        "required": ["cell_ref", "digit"]
    }   
}

find_identical_candidates_peers_schema = {
    "name": "find_identical_candidates_peers_schema_fn",
    "description": "Returns all the peer cells (excluding the current cell) that have exactly the same candidate list and candidate digits as the current cell. Useful for checking naked pairs/triples/quads.",
    "parameters": {
        "type": "object",
        "properties": {
            "cell_ref": {
                "type": "string",
                "description": "The cell reference (e.g. 'R1C1')."
            }
        },
        "required": ["cell_ref"]
    }
}

find_subset_candidates_peers_schema = {
    "name": "find_subset_candidates_peers_schema_fn",
    "description": "Returns all the peer cells that have candidate lists that are a subset of the specified candidate list. For example if the specified candidate list is {1, 2, 3}, this will pick out cells that have candidate lists like {1, 2}, {1, 3}, {2, 3}, {1, 2, 3}. Useful for finding hidden and naked pairs/triples/quads.",
    "parameters": {
        "type": "object",
        "properties": {
            "cell_ref": {
                "type": "string",
                "description": "The cell reference (e.g. 'R1C1')."
            },
            "candidate_list": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "The candidate list containing the candidate digits to compare against."
            }
        },
        "required": ["cell_ref", "candidate_list"]
    }
}

# Put them all in a list so that the LLM knows what is available.
sudoku_function_schemas = [
    get_cell_contents_schema,
    find_assigned_peer_schema,
    find_candidate_peers_schema,
    find_identical_candidates_peers_schema,
    find_subset_candidates_peers_schema
]


# -----------------------------------------
# LLM Agentic Function Wrappers
# -----------------------------------------
#    In order to allow the LLM to call the original function, the arguments need to be extracted
#    and parsed into the correct type, before passing them on to the actual function call.
#    The result of the original function also needs to be wrapped into a dict to be passed back to the LLM.

def get_cell_contents_schema_fn(puzzle: Dict[str, dict], arguments:dict) -> dict:
    """
    LLM Function wrapper for get_cell_contents.
    """
    cell_ref = arguments['cell_ref']

    try:
        cell_contents = get_cell_contents(puzzle=puzzle, cell_ref=cell_ref)
    except Exception as e:
        print(f"CRITICAL: get_cell_contents_as_string - Error: {e}")
        return {"error": str(e)}
    
    return {cell_ref: cell_contents}

def find_assigned_peer_schema_fn(puzzle: Dict[str, dict], arguments:dict) -> dict:
    """
    LLM Function wrapper for find_assigned_peer.
    """
    cell_ref = arguments.get('cell_ref')
    digit = arguments.get('digit')

    try:
        peer = find_assigned_peer(puzzle=puzzle, cell_ref=cell_ref, digit=digit)
    except Exception as e:
        print(f"CRITICAL: find_assigned_peer - Error: {e}")
        return {"error": str(e)}
    
    return {"cell_ref": peer}

def find_candidate_peers_schema_fn(puzzle: Dict[str, dict], arguments:dict) -> dict:
    """
    LLM Function wrapper for find_candidate_peers.
    """
    cell_ref = arguments.get('cell_ref')
    digit = arguments.get('digit')

    try:
        peers = find_candidate_peers(puzzle=puzzle, cell_ref=cell_ref, digit=digit)
    except Exception as e:
        print(f"CRITICAL: find_candidate_peers - Error: {e}")
        return {"error": str(e)}
    
    return {"peers": peers}

def find_identical_candidates_peers_schema_fn(puzzle: Dict[str, dict], arguments:dict) -> dict:
    """
    LLM Function wrapper for find_identical_candidates_peers.
    """
    cell_ref = arguments.get('cell_ref')

    try:
        peers = find_identical_candidates_peers(puzzle=puzzle, cell_ref=cell_ref)
    except Exception as e:
        print(f"CRITICAL: find_identical_candidates_peers - Error: {e}")
        return {"error": str(e)}
    
    return {"peers": peers}

def find_subset_candidates_peers_schema_fn(puzzle: Dict[str, dict], arguments:dict) -> dict:
    """
    LLM Function wrapper for find_subset_candidates_peers.
    """
    cell_ref = arguments.get('cell_ref')
    candidate_list_str = arguments.get('candidate_list')

    # Attempt to convert to integer list.
    try:
        numbers = re.findall(r'-?\d+', candidate_list_str)
        candidate_list = [int(num) for num in numbers]
    except (ValueError, TypeError) as e:
        print(f"CRITICAL: find_subset_candidates_peers_schema_fn - Error: {e}")
        return {"error": f"Invalid input data: {e}"}
    
    try:
        peers = find_subset_candidates_peers(puzzle=puzzle, cell_ref=cell_ref, candidate_list=candidate_list)
    except Exception as e:
        print(f"CRITICAL: find_subset_candidates_peers - Error: {e}")
        return {"error": str(e)}
    
    return {"peers": peers}

# -----------------------------------------
# LLM Calls (OpenAI)
# -----------------------------------------
def propose_next_move(puzzle: Dict[str, dict]) -> NextMove:
    """
    Given a puzzle (JSON/dict) representing the sudoku board, call the OpenAI LLM
    to propose the next move. The function returns a NextMove instance.
    """
    # Convert the puzzle to a text representation
    puzzle_str = json.dumps(puzzle)

    # In-memory conversation store
    conversation_history = []

    # Call the LLM.
    assistant_message = call_llm(puzzle_str, conversation_history)

    # Check if assistant wants to call a function
    while assistant_message.function_call:
        function_name = assistant_message.function_call.name
        arguments = json.loads(assistant_message.function_call.arguments)

        # Execute the function and add the response to the conversation.
        if function_name:
            print(f"INFO: Calling function: \"{function_name}\" with arguments: {arguments}")
    
            if function_name == "get_cell_contents_schema_fn":
                function_response = get_cell_contents_schema_fn(puzzle, arguments)
            elif function_name == "find_assigned_peer_schema_fn":
                function_response = find_assigned_peer_schema_fn(puzzle, arguments)
            elif function_name == "find_candidate_peers_schema_fn":
                function_response = find_candidate_peers_schema_fn(puzzle, arguments)
            elif function_name == "find_identical_candidates_peers_schema_fn":
                function_response = find_identical_candidates_peers_schema_fn(puzzle, arguments)
            elif function_name == "find_subset_candidates_peers_schema_fn":
                function_response = find_subset_candidates_peers_schema_fn(puzzle, arguments)

        if function_response.get("error"):
            print(f"CRITICAL: propose_next_move - Function {function_name} error: {function_response.get('error')}")
            return
        
        print(f"INFO:   Function Reply: {function_response}")
        
        conversation_history.append({
            "role": "function",
            "name": function_name,
            "content": json.dumps(function_response),
        })
    
        # Continue the conversation
        assistant_message = call_llm(puzzle_str, conversation_history)
    
    # Ensure there is content for the assistant's message.  
    if assistant_message.content is None:
        assistant_message.content = "I am sorry, I am not able to process your request."
    else:
        # Extract the LLM's reply
        answer = strip_pre_post(
            assistant_message.content.strip()
        )

        # print(f"INFO: LLM Response: {answer}\n")

        try:
            data = json.loads(answer)
            next_move = NextMove(**data)
            return next_move
        except Exception as e:
            raise ValueError("CRITICAL: Failed to parse LLM response: " + str(e))

def call_llm(puzzle_str: str, conversation_history: List[str]):
    # Create a prompt for the LLM.
    system_prompt = (
        "You are an expert sudoku solving agent. Your task is to analyze the current puzzle board and propose the next move without modifying the board.\n"
#        "Use the helper functions (whose schemas are provided) to inspect the board and verify your answer. Do not assume any digit is in a cell without checking with these functions.\n\n"
        "Output only a JSON array containing one object with the keys:\n"
        "\"strategy\": the name of the solving technique used\n"
        "\"reasoning\": a detailed explanation of your reasoning\n"
        "\"steps\": a list of steps to achieve the strategy\n"
        "\"cell\": the cell reference (e.g. 'R1C2')\n"
        "\"action\": either 'assign' or 'eliminate'\n"
        "\"digit\": the digit to assign or eliminate\n\n"
#        "Available helper functions:\n"
#        + "\n".join([f"{func['name']}: {func['description']}" for func in sudoku_function_schemas])
    )

    # print(f"INFO: Puzzle_str\n{puzzle_str}")

    user_prompt = (
        "Below is the current 9x9 Sudoku board represented as JSON string.\n"
        "Each cell is referenced by a cell reference \"RxCy\" which denotes Row x and Column y.\n"
        f"{puzzle_str}\n\n"
        "A solved cell is represented by its digit under the key \"value\".\n"
        "An unsolved cell has null for value but has a candidate list under the key \"candidates\", which contains a list of the possible digits that can be likely for this cell.\n"
        "Analyze the board and propose the next move based solely on the data provided.\n\n"
        "Output only a JSON array:\n"
        "[\n"
        "  {\n"
        "    \"strategy\": \"xxxx\",\n"
        "    \"reasoning\": \"xxxx\",\n" 
        "    \"steps\": [\n"
        "      { \"cell\": \"RxCy\",\n"
        "        \"action\": \"'assign' or 'eliminate'\",\n"
        "        \"digit\": x\n"
        "      },\n"
        "      { ... }\n"
        "  }\n"
        "]\n"
    )

    # Call the OpenAI Chat API
    try:
        response = client.chat.completions.create(
            messages = [
                {"role": "assistant", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ] + conversation_history,
            model = "o1-preview",
#            functions = sudoku_function_schemas,  # Pass the function schemas.
#            function_call = "auto"         # Let the API decide whether to call a function.
        )
    except Exception as e:
        raise Exception("OpenAI API call failed: " + str(e))

    return response.choices[0].message