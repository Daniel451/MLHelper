import os
import shutil


def clean_directory(path: str, rm_subdirs: bool = False):
    """
    deletes all files in given path after confirmation in terminal

    :param path: path to clean
    """
    print()
    answer = None
    while answer != "y" and answer != "n":
        answer = input(f"should path '{path}' be cleaned now? [y/n]: ")

    if answer == "y":
        force_clean_directory(path, rm_subdirs=rm_subdirs)


def force_clean_directory(path, rm_subdirs: bool = False):
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
        for e in os.listdir(path):
            p = os.path.join(path, e)
            # file
            if os.path.isfile(p):
                print(f"deleting '{p}'")
                os.unlink(p)
            # dir
            elif os.path.isdir(p) and rm_subdirs:
                shutil.rmtree(p, ignore_errors=True)

        print(f"path '{path}' is clean now.")
    else:
        print(f"path '{path}' seems to be empty.")
    print()

