import pandas as pd
from collections import deque, defaultdict


def load_data(directory):
    """
    Load and parse CSV files into data structures.
    Returns:
        name_to_id: Dictionary mapping scientist names to their IDs.
        scientist_info: Dictionary mapping scientist IDs to their name and authored papers.
        paper_info: Dictionary mapping paper IDs to their title, year, and authors.
    """
    try:
        scientists = pd.read_csv(f"{directory}/scientists.csv")
        papers = pd.read_csv(f"{directory}/papers.csv")
        authors = pd.read_csv(f"{directory}/authors.csv")
    except FileNotFoundError:
        print(f"Error: Directory '{directory}' or required CSV files not found.")
        return None, None, None

    # Initialize data structures
    name_to_id = dict(zip(scientists['name'], scientists['scientist_id']))
    scientist_info = defaultdict(lambda: {'name': '', 'papers': set()})
    paper_info = defaultdict(lambda: {'title': '', 'year': 0, 'authors': set()})

    # Populate scientist_info
    for _, row in scientists.iterrows():
        scientist_info[row['scientist_id']]['name'] = row['name']

    # Populate paper_info
    for _, row in papers.iterrows():
        paper_info[row['paper_id']]['title'] = row['title']
        paper_info[row['paper_id']]['year'] = row['year']

    # Populate authorship relationships
    for _, row in authors.iterrows():
        scientist_info[row['scientist_id']]['papers'].add(row['paper_id'])
        paper_info[row['paper_id']]['authors'].add(row['scientist_id'])

    return name_to_id, scientist_info, paper_info


def neighbors_for_person(scientist_id, scientist_info, paper_info):
    """
    Return a set of (paper_id, scientist_id) pairs for all co-authors of the given scientist.
    """
    neighbors = set()
    for paper_id in scientist_info[scientist_id]['papers']:
        for co_author_id in paper_info[paper_id]['authors']:
            if co_author_id != scientist_id:
                neighbors.add((paper_id, co_author_id))
    return neighbors


def shortest_path(source_id, target_id, scientist_info, paper_info):
    """
    Find the shortest path from source_id to target_id using BFS.
    Returns:
        A list of (paper_id, scientist_id) tuples representing the path, or None if no path exists.
    """
    if source_id == target_id:
        return []

    frontier = deque([{'scientist': source_id, 'path': []}])
    explored = set([source_id])

    while frontier:
        node = frontier.popleft()
        current_scientist = node['scientist']
        current_path = node['path']

        for paper_id, next_scientist in neighbors_for_person(current_scientist, scientist_info, paper_info):
            if next_scientist not in explored:
                new_path = current_path + [(paper_id, next_scientist)]
                if next_scientist == target_id:
                    return new_path
                explored.add(next_scientist)
                frontier.append({'scientist': next_scientist, 'path': new_path})

    return None


def display_path(path, scientist_info, paper_info, source_name, target_name):
    """
    Display the degrees of separation and the connecting papers.
    """
    if path is None:
        print(f"No path exists between {source_name} and {target_name}.")
        return
    degrees = len(path)
    print(f"{degrees} degrees of separation.")
    for i, (paper_id, scientist_id) in enumerate(path, 1):
        prev_scientist = source_name if i == 1 else scientist_info[path[i - 2][1]]['name']
        curr_scientist = scientist_info[scientist_id]['name']
        title = paper_info[paper_id]['title']
        print(f"{i}: {prev_scientist} and {curr_scientist} co-authored \"{title}\"")


def main():
    """
    Main function to run the program.
    Usage: python degrees.py directory
    Prompts for two scientist names and displays the shortest path.
    """
    import sys
    if len(sys.argv) != 2:
        print("Usage: python degrees.py directory")
        sys.exit(1)

    directory = sys.argv[1]
    print("Loading data...")
    name_to_id, scientist_info, paper_info = load_data(directory)

    if name_to_id is None:
        sys.exit(1)

    print("Data loaded.")

    source_name = input("Name: ").strip()
    target_name = input("Name: ").strip()

    source_id = name_to_id.get(source_name)
    target_id = name_to_id.get(target_name)

    if not source_id or not target_id:
        print("Error: One or both scientist names are invalid.")
        sys.exit(1)

    path = shortest_path(source_id, target_id, scientist_info, paper_info)
    display_path(path, scientist_info, paper_info, source_name, target_name)


if __name__ == "__main__":
    main()