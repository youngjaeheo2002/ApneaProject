PATH = PATH = '/scratch/alim/overnight_validation/ANNE-PSG231215'
def get_Paths_and_Names(path = PATH):
    paths_and_names = []
    parent_dir = Path(path)

    for subDir in parent_dir.iterdir():
        if subDir.is_dir():
            for entry in subDir.iterdir():
                absolute_entry_path = str(entry)
                paths_and_names.append({
                    'path': absolute_entry_path,
                    'name':subDir.name
                })


    return paths_and_names