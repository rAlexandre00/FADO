import hashlib


__all__ = ['file_changed', 'write_file_hash']


def file_changed(file_path, hash_path):
    """Checks if a file has changed

        Parameters:
            file_path(str): file to be checked
            hash_path(str): path of the hash that was done on a previous file version

        Returns:
            True if file has changed or hash does not exist
    """
    file_hash = get_file_hash(file_path)
    try:
        with open(hash_path, 'r') as f:
            return file_hash != f.readline()
    except FileNotFoundError:
        return True


def write_file_hash(file_path, hash_path):
    """Creates a file with the hash of a given file

        Parameters:
            file_path(str): file to be hashed
            hash_path(str): output path of the hash

    """
    file_hash = get_file_hash(file_path)
    with open(hash_path, 'w') as f:
        f.write(file_hash)


def get_file_hash(file_path):
    """Calculates the hash of a file

        Parameters:
            file_path(str): file to calculate hash from

        Returns:
            sha1 hash of the file in hexadecimal

    """
    buf_size = 65536  # lets read stuff in 64kb chunks!

    sha1 = hashlib.sha1()

    with open(file_path, 'rb') as f:
        while True:
            data = f.read(buf_size)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest()
