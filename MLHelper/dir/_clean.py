import os


def clean_directory(path: str):
    """
    deletes all files in given path after confirmation in terminal

    :param path: path to clean
    """
    print()
    answer = None
    while answer != "y" and answer != "n":
        answer = input(f"should path '{path}' be cleaned now? [y/n]: ")

    if answer == "y":
        force_clean_directory(path)


def force_clean_directory(path):
    """
    deletes all files in given path without asking the user for confirmation

    :param path: path to clean
    """
    if not os.path.exists(path):
        print(f"path '{path}' does not exist.")
        return

    print()
    print(f"cleaning path '{path}'...")
    if len(os.listdir(path)) > 0:
        for f in os.listdir(path):
            fpath = os.path.join(path, f)
            if os.path.isfile(fpath):
                print(f"deleting '{fpath}'")
                os.unlink(fpath)
        print(f"path '{path}' is clean now.")
    else:
        print(f"path '{path}' seems to be empty.")
    print()

