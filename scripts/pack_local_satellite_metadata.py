"""Reduce local disk allocation for archived metadata or research indexes without changing their paths."""
import argparse
import ctypes
import hashlib
import itertools
import json
import os
import stat
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from collect_public import RAW, ROOT

COMPRESSED_FLAG = 0x20  # UF_COMPRESSED in Darwin sys/stat.h
CREATION_ATTR = struct.pack("=HHIIIII", 5, 0, 0x00000200, 0, 0, 0, 0)  # ATTR_CMN_CRTIME
NOFOLLOW = 1


def creation_api():
    libc = ctypes.CDLL(None, use_errno=True)
    for function in (libc.getattrlist, libc.setattrlist):
        function.argtypes = (ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong)
        function.restype = ctypes.c_int
    return libc


def creation_time(libc, path):
    buffer = ctypes.create_string_buffer(20)  # length prefix and a Darwin timespec
    if libc.getattrlist(os.fsencode(path), CREATION_ATTR, buffer, len(buffer), NOFOLLOW):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), path)
    if struct.unpack_from("=I", buffer)[0] != len(buffer):
        raise ValueError(f"Unexpected creation time record for {path}")
    return bytes(buffer[4:])


def extended_attributes(path):
    return subprocess.check_output(["/usr/bin/xattr", "-l", str(path)])


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def restore_times(libc, path, original, created):
    if libc.setattrlist(os.fsencode(path), CREATION_ATTR, created, len(created), NOFOLLOW):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), path)
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))


def pack(libc, path):
    original = path.lstat()
    if (not stat.S_ISREG(original.st_mode) or original.st_nlink != 1 or original.st_size <= 4096
            or original.st_flags & COMPRESSED_FLAG):
        return None
    created = creation_time(libc, path)
    with tempfile.TemporaryDirectory(prefix=".metadata-pack-", dir=path.parent) as work:
        output = Path(work) / path.name
        subprocess.run(["/usr/bin/ditto", "--hfsCompression", str(path), str(output)], check=True)
        packed = output.stat()
        if packed.st_blocks >= original.st_blocks:
            return None
        checksum = sha256(path)
        if sha256(output) != checksum or extended_attributes(path) != extended_attributes(output):
            raise ValueError(f"Packed copy did not preserve source bytes and attributes: {path}")
        restore_times(libc, output, original, created)
        packed = output.stat()
        fields = ("st_size", "st_mtime_ns", "st_mode", "st_uid", "st_gid")
        if (any(getattr(packed, name) != getattr(original, name) for name in fields)
                or creation_time(libc, output) != created or (packed.st_flags & ~COMPRESSED_FLAG) != original.st_flags):
            raise ValueError(f"Packed copy did not preserve source file metadata: {path}")
        current = path.lstat()
        if (current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns) != (
                original.st_ino, original.st_size, original.st_mtime_ns, original.st_ctime_ns):
            raise ValueError(f"Source changed while its copy was verified: {path}")
        output.replace(path)
    return {"path": str(path.relative_to(ROOT)), "sha256": checksum,
            "source_bytes": original.st_size, "before_allocated_bytes": original.st_blocks * 512,
            "after_allocated_bytes": packed.st_blocks * 512}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--derived-index", action="store_true", help="Pack the built SQLite index and top-level CSV exports")
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("Transparent packing is only supported on macOS")
    libc = creation_api()
    if args.derived_index:
        derived = ROOT / "data" / "derived"
        paths = [derived / "cushing.sqlite", *sorted(derived.glob("*.csv"))]
        log_name = "local_index_packing.jsonl"
    else:
        l1c = RAW / "sentinel2_l1c_original"
        roots = (RAW / "sentinel2_metadata", l1c / "granules", l1c / "supplemental" / "granules")
        paths = (path for root in roots for path in sorted(root.rglob("*")) if path.suffix in {".xml", ".gml"})
        log_name = "local_metadata_packing.jsonl"
    count, saved = 0, 0
    with (ROOT / "data" / "logs" / log_name).open("a", buffering=1) as log:
        for path in itertools.islice(paths, args.limit):
            record = pack(libc, path)
            if record is not None:
                print(json.dumps(record), file=log)
                count += 1
                saved += record["before_allocated_bytes"] - record["after_allocated_bytes"]
    print(f"{count} files packed, {saved / 1024 ** 3:.3f} GiB of allocation saved")


if __name__ == "__main__":
    main()
