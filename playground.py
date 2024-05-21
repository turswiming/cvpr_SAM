import os
def getBaseNameWithoutExtension(path: str) -> str:
    name_with_extension: str = os.path.basename(path)
    name_list: list[str] = os.path.splitext(name_with_extension)
    #remove the final one in list
    return ".".join(name_list)

val = getBaseNameWithoutExtension("playgro.und.py") # playground
print(val)