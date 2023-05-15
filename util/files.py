from os.path import splitext


def get_ext(filename: str) -> str:
    split = splitext(filename)
    file_extension = split[1]
    return file_extension


def is_img(filename: str) -> bool:
    if not isinstance(filename, str):
        raise ValueError(f"{filename} is not a string.")

    image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".gif"]
    extension = splitext(filename)[1].lower()
    return extension in image_extensions
