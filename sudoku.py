#!/usr/bin/env python3
"""
Sudoku Game - CLI-based terminal game
"""

# Hardcoded Sudoku puzzles at 3 difficulty levels
PUZZLES = {
    "easy": [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ],
    "medium": [
        [0, 0, 0, 6, 0, 0, 4, 0, 0],
        [7, 0, 0, 0, 0, 3, 6, 0, 0],
        [0, 0, 0, 0, 9, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 5, 0, 1, 8, 0, 0, 0, 0],
        [3, 0, 0, 0, 0, 0, 0, 0, 4],
        [0, 0, 2, 3, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    "hard": [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 3, 0, 8, 5],
        [0, 0, 1, 0, 2, 0, 0, 0, 0],
        [0, 0, 0, 5, 0, 7, 0, 0, 0],
        [0, 0, 4, 0, 0, 0, 1, 0, 0],
        [0, 9, 0, 0, 0, 0, 0, 0, 0],
        [5, 0, 0, 0, 0, 0, 0, 7, 3],
        [0, 0, 2, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 0, 0, 0, 9],
    ],
}


def print_board(board):
    """Print the Sudoku board with visual distinction."""
    print("\n    1 2 3   4 5 6   7 8 9")
    print("  +-------+-------+-------+")
    for i in range(9):
        row = ""
        for j in range(9):
            if j == 0:
                row += f"{i+1} | "
            elif j == 3 or j == 6:
                row += "| "
            
            cell = str(board[i][j]) if board[i][j] != 0 else "."
            row += cell + " "
            
            if j == 8:
                row += "|"
        print(row)
        if i == 2 or i == 5:
            print("  +-------+-------+-------+")
    print("  +-------+-------+-------+")


def is_valid_placement(board, row, col, num):
    """Check if placing num at board[row][col] is valid."""
    # Check row
    for c in range(9):
        if board[row][c] == num:
            return False, f"Invalid: {num} already in row {row+1}"
    
    # Check column
    for r in range(9):
        if board[r][col] == num:
            return False, f"Invalid: {num} already in column {col+1}"
    
    # Check 3x3 box
    box_row, box_col = (row // 3) * 3, (col // 3) * 3
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if board[r][c] == num:
                return False, f"Invalid: {num} already in 3x3 box"
    
    return True, ""


def is_board_complete(board):
    """Check if board is completely filled."""
    for row in board:
        for cell in row:
            if cell == 0:
                return False
    return True


def is_board_valid(board):
    """Check if board is valid (no duplicates in rows, cols, boxes)."""
    # Check rows
    for r in range(9):
        seen = set()
        for c in range(9):
            if board[r][c] != 0:
                if board[r][c] in seen:
                    return False
                seen.add(board[r][c])
    
    # Check columns
    for c in range(9):
        seen = set()
        for r in range(9):
            if board[r][c] != 0:
                if board[r][c] in seen:
                    return False
                seen.add(board[r][c])
    
    # Check boxes
    for box_r in range(0, 9, 3):
        for box_c in range(0, 9, 3):
            seen = set()
            for r in range(box_r, box_r + 3):
                for c in range(box_c, box_c + 3):
                    if board[r][c] != 0:
                        if board[r][c] in seen:
                            return False
                        seen.add(board[r][c])
    
    return True


def get_fixed_cells(puzzle):
    """Return a set of (row, col) tuples representing fixed cells."""
    fixed = set()
    for r in range(9):
        for c in range(9):
            if puzzle[r][c] != 0:
                fixed.add((r, c))
    return fixed


def copy_board(board):
    """Create a deep copy of the board."""
    return [row[:] for row in board]


def select_difficulty():
    """Let user select difficulty level."""
    print("\n=== Select Difficulty ===")
    print("1. Easy")
    print("2. Medium")
    print("3. Hard")
    
    while True:
        choice = input("\nEnter choice (1-3): ").strip()
        if choice == "1":
            return "easy"
        elif choice == "2":
            return "medium"
        elif choice == "3":
            return "hard"
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


def parse_input(user_input):
    """Parse user input in format 'row col value' or 'row col' to clear."""
    parts = user_input.split()
    
    if len(parts) == 2:
        # Clear cell: "row col" (value defaults to 0)
        try:
            row = int(parts[0]) - 1  # Convert to 0-based
            col = int(parts[1]) - 1
            value = 0
        except ValueError:
            return None, None, None, "Invalid format. Use 'row col value' or 'row col' to clear."
    elif len(parts) == 3:
        try:
            row = int(parts[0]) - 1  # Convert to 0-based
            col = int(parts[1]) - 1
            value = int(parts[2])
        except ValueError:
            return None, None, None, "Invalid format. Use 'row col value' or 'row col' to clear."
    else:
        return None, None, None, "Invalid format. Use 'row col value' or 'row col' to clear."
    
    # Validate ranges
    if not (0 <= row <= 8):
        return None, None, None, "Invalid row. Use 1-9."
    if not (0 <= col <= 8):
        return None, None, None, "Invalid column. Use 1-9."
    if value != 0 and not (1 <= value <= 9):
        return None, None, None, "Invalid value. Use 1-9 or 0 to clear."
    
    return row, col, value, None


def play_game():
    """Main game loop."""
    print("=" * 40)
    print("       WELCOME TO SUDOKU!")
    print("=" * 40)
    print("\nHow to play:")
    print("  - Enter 'row col value' to place a number")
    print("  - Enter 'row col' to clear a cell")
    print("  - Use 1-9 for rows, columns, and values")
    print("  - Enter 'q' to quit")
    print("  - Enter 'n' for a new game")
    print("  - Fixed cells (given numbers) cannot be changed")
    
    # Select difficulty
    difficulty = select_difficulty()
    puzzle = PUZZLES[difficulty]
    board = copy_board(puzzle)
    fixed_cells = get_fixed_cells(puzzle)
    
    while True:
        print_board(board)
        
        # Check win condition
        if is_board_complete(board):
            if is_board_valid(board):
                print("\n" + "=" * 40)
                print("  CONGRATULATIONS! YOU WIN!")
                print("=" * 40)
                break
            else:
                print("\nError: Board is full but invalid!")
                print("Fix the errors and try again.")
        
        # Get user input
        user_input = input("\nEnter move (row col [value]): ").strip().lower()
        
        if user_input == "q":
            print("\nThanks for playing! Goodbye!")
            break
        
        if user_input == "n":
            print("\nStarting new game...")
            difficulty = select_difficulty()
            puzzle = PUZZLES[difficulty]
            board = copy_board(puzzle)
            fixed_cells = get_fixed_cells(puzzle)
            continue
        
        if user_input == "h" or user_input == "help":
            print("\nCommands:")
            print("  row col value  - Place a number (e.g., '1 3 5')")
            print("  row col        - Clear a cell (e.g., '1 3')")
            print("  n              - New game")
            print("  q              - Quit")
            print("  h              - Show this help")
            continue
        
        row, col, value, error = parse_input(user_input)
        
        if error:
            print(f"\nError: {error}")
            continue
        
        # Check if trying to modify fixed cell
        if (row, col) in fixed_cells and value == 0:
            print(f"\nError: Cannot clear fixed cell at row {row+1}, col {col+1}")
            continue
        
        if (row, col) in fixed_cells:
            print(f"\nError: Cannot modify fixed cell at row {row+1}, col {col+1}")
            continue
        
        # Validate placement
        if value != 0:
            valid, error_msg = is_valid_placement(board, row, col, value)
            if not valid:
                print(f"\n{error_msg}")
                continue
        
        # Place the value
        board[row][col] = value
        print(f"\nPlaced {value if value != 0 else '(empty)'} at row {row+1}, col {col+1}")


if __name__ == "__main__":
    play_game()
