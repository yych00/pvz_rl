"""Decode native JSON, PVZB v1 facts, or a Windows shared-result descriptor."""
from __future__ import annotations
import json
import mmap
import struct
from collections.abc import Mapping


def decode_payload(payload):
    if isinstance(payload, Mapping):
        return payload
    if isinstance(payload, str):
        return json.loads(payload)
    if payload.startswith(b"PVZM"):
        descriptor = json.loads(payload[4:])
        size, length, revision = (int(descriptor[n]) for n in ("size", "length", "revision"))
        name = descriptor["name"]
        if not isinstance(name, str) or not name.startswith("Local\\pvz-gym-") or not 12 <= size <= 16*1024*1024 or not 0 < length <= size-12:
            raise ValueError("invalid shared runtime descriptor")
        with mmap.mmap(-1, size, tagname=name, access=mmap.ACCESS_READ) as shared:
            before = struct.unpack_from("<QI", shared, 0)
            payload = bytes(shared[12:12+length])
            after = struct.unpack_from("<QI", shared, 0)
        if before != (revision, length) or after != before:
            raise ValueError("shared runtime result changed during read")
    if not payload.startswith(b"PVZB\x01"):
        return json.loads(payload)
    view = memoryview(payload)
    offset = 5

    def take(fmt):
        nonlocal offset
        size = struct.calcsize(fmt)
        if offset + size > len(view):
            raise ValueError("truncated binary runtime result")
        value = struct.unpack_from(fmt, view, offset)[0]
        offset += size
        return value

    def read(depth=0):
        nonlocal offset
        if depth > 32:
            raise ValueError("binary runtime nesting exceeds limit")
        tag = take("<B")
        if tag == 0: return None
        if tag in (1, 2): return tag == 2
        if tag == 3: return take("<q")
        if tag == 4: return take("<d")
        if tag == 5:
            length = take("<I")
            if offset + length > len(view): raise ValueError("truncated binary runtime string")
            value = bytes(view[offset:offset+length]).decode("utf-8")
            offset += length
            return value
        if tag in (6, 7):
            length = take("<I")
            if length > len(view)-offset: raise ValueError("invalid binary runtime collection length")
            if tag == 6: return [read(depth+1) for _ in range(length)]
            result = {}
            for _ in range(length):
                key = read(depth+1)
                if not isinstance(key, str) or key in result: raise ValueError("invalid binary runtime key")
                result[key] = read(depth+1)
            return result
        raise ValueError(f"unknown binary runtime tag {tag}")

    result = read()
    if offset != len(view): raise ValueError("trailing binary runtime bytes")
    return result
