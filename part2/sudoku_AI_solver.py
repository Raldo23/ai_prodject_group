from collections import defaultdict, deque
import copy

class SudokuAISolver:
    def __init__(self, grid):
        self.grid = grid
        self.variables = [(i, j) for i in range(9) for j in range(9)]
        self.domains = {(i, j): {1, 2, 3, 4, 5, 6, 7, 8, 9} if grid[i][j] == 0 else {grid[i][j]} for i, j in self.variables}
        self.neighbors = self._get_neighbors()
        self.enforce_node_consistency()

    def _get_neighbors(self):
        neighbors = defaultdict(list)
        for i, j in self.variables:
            for r in range(9):
                if r != i:
                    neighbors[(i, j)].append((r, j))
            for c in range(9):
                if c != j:
                    neighbors[(i, j)].append((i, c))
            box_row, box_col = 3 * (i // 3), 3 * (j // 3)
            for r in range(box_row, box_row + 3):
                for c in range(box_col, box_col + 3):
                    if (r, c) != (i, j) and (r, c) not in neighbors[(i, j)]:
                        neighbors[(i, j)].append((r, c))
        return neighbors

    def enforce_node_consistency(self):
        for i, j in self.variables:
            if self.grid[i][j] != 0:
                self.domains[(i, j)] = {self.grid[i][j]}

    def revise(self, xi, xj):
        revised = False
        values_to_remove = set()
        for x in self.domains[xi]:
            if not any(y in self.domains[xj] for y in self.domains[xj] if x != y):
                values_to_remove.add(x)
                revised = True
        self.domains[xi] -= values_to_remove
        return revised

    def ac3(self):
        queue = deque([(xi, xj) for xi in self.variables for xj in self.neighbors[xi]])
        while queue:
            xi, xj = queue.popleft()
            if self.revise(xi, xj):
                if not self.domains[xi]:
                    return False
                for xk in self.neighbors[xi]:
                    if xk != xj:
                        queue.append((xk, xi))
        return True

    def assignment_complete(self, assignment):
        return all((i, j) in assignment for i, j in self.variables)

    def consistent(self, assignment):
        for (i, j), value in assignment.items():
            for r in range(9):
                if r != i and (r, j) in assignment and assignment[(r, j)] == value:
                    return False
            for c in range(9):
                if c != j and (i, c) in assignment and assignment[(i, c)] == value:
                    return False
            box_row, box_col = 3 * (i // 3), 3 * (j // 3)
            for r in range(box_row, box_row + 3):
                for c in range(box_col, box_col + 3):
                    if (r, c) != (i, j) and (r, c) in assignment and assignment[(r, c)] == value:
                        return False
        return True

    def order_domain_values(self, var, assignment):
        return sorted(self.domains[var], key=lambda val: sum(1 for neighbor in self.neighbors[var] if neighbor not in assignment and val in self.domains[neighbor]))

    def select_unassigned_variable(self, assignment):
        unassigned = [var for var in self.variables if var not in assignment]
        return min(unassigned, key=lambda var: (len(self.domains[var]), -sum(1 for neighbor in self.neighbors[var] if neighbor not in assignment)))

    def backtrack(self, assignment):
        if self.assignment_complete(assignment):
            return assignment
        var = self.select_unassigned_variable(assignment)
        for value in self.order_domain_values(var, assignment):
            assignment[var] = value
            if self.consistent(assignment):
                old_domains = copy.deepcopy(self.domains)
                inferences = self._infer(var, value)
                if inferences is not None:
                    result = self.backtrack(assignment)
                    if result is not None:
                        return result
                self.domains = old_domains
            del assignment[var]
        return None

    def _infer(self, var, value):
        self.domains[var] = {value}
        if not self.ac3():
            return None
        return self.domains

def read_puzzle(filename):
    grid = []
    with open(filename, 'r') as f:
        for line in f:
            grid.append([int(x) for x in line.strip().split()])
    return grid

def print_grid(grid):
    for row in grid:
        print(' '.join(str(x) for x in row))

def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: python sudoku_AI_solver.py puzzle.txt")
        return
    grid = read_puzzle(sys.argv[1])
    solver = SudokuAISolver(grid)
    assignment = solver.backtrack({})
    if assignment:
        for i, j in solver.variables:
            grid[i][j] = assignment[(i, j)]
        print("Solved Sudoku:")
        print_grid(grid)
    else:
        print("No solution exists.")

if __name__ == "__main__":
    main()