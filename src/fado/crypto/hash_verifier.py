import sys
import hashlib


def file_changed(file_path, hash_path):
    file_hash = get_file_hash(file_path)
    try:
        with open(hash_path, 'r') as f:
            return file_hash != f.readline()
    except FileNotFoundError:
        return True


def write_file_hash(file_path, hash_path):
    file_hash = get_file_hash(file_path)
    with open(hash_path, 'w') as f:
        f.write(file_hash)


def get_file_hash(file_path):
    buf_size = 65536  # lets read stuff in 64kb chunks!

    sha1 = hashlib.sha1()

    with open(file_path, 'rb') as f:
        while True:
            data = f.read(buf_size)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest()
