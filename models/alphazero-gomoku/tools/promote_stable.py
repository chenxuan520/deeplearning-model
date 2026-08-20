#!/usr/bin/env python3
"""Promote one immutable .net file to the fixed AlphaZero stable channel."""

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parents[1]
PUBLIC_DIR = MODEL_DIR / "public"
CHANNEL_FILE = PUBLIC_DIR / "channels" / "stable.json"
PUBLIC_BASE = "https://azgomoku.011203.xyz"


def parse_header_bytes(data):
    if len(data) != 64 or data[:8] != b"XQPVRN01":
        raise ValueError("not an XQPVRN01 model")
    values = struct.unpack("<I11i2f", data[8:])
    version = values[0]
    if version != 1:
        raise ValueError("unsupported model version: %d" % version)
    keys = [
        "input_channels", "board_height", "board_width", "trunk_channels",
        "residual_blocks", "policy_channels", "policy_size",
        "value_channels", "value_hidden_dim", "thread_num", "rand_seed",
    ]
    config = dict(zip(keys, values[1:12]))
    config["batch_norm_epsilon"] = values[12]
    config["batch_norm_momentum"] = values[13]
    return config


def expected_vector_lengths(config):
    trunk = config["trunk_channels"]
    input_channels = config["input_channels"]
    blocks = config["residual_blocks"]
    policy_channels = config["policy_channels"]
    policy_size = config["policy_size"]
    value_channels = config["value_channels"]
    value_hidden = config["value_hidden_dim"]
    cells = config["board_height"] * config["board_width"]

    def conv(out_channels, in_channels, kernel):
        return [out_channels * in_channels * kernel * kernel, out_channels]

    def batch_norm(channels):
        return [channels, channels, channels, channels]

    lengths = conv(trunk, input_channels, 3) + batch_norm(trunk)
    for _ in range(blocks):
        lengths += conv(trunk, trunk, 3) + batch_norm(trunk)
        lengths += conv(trunk, trunk, 3) + batch_norm(trunk)
    lengths += conv(policy_channels, trunk, 1) + batch_norm(policy_channels)
    lengths += [policy_size * policy_channels * cells, policy_size]
    lengths += conv(value_channels, trunk, 1) + batch_norm(value_channels)
    lengths += [value_hidden * value_channels * cells, value_hidden]
    lengths += [value_hidden, 1]
    if len(lengths) != 72:
        raise ValueError("internal tensor layout error")
    return lengths


def validate_model(path):
    with path.open("rb") as source:
        header = source.read(64)
        config = parse_header_bytes(header)
        for index, expected in enumerate(expected_vector_lengths(config)):
            prefix = source.read(8)
            if len(prefix) != 8:
                raise ValueError("truncated tensor prefix at index %d" % index)
            count = struct.unpack("<Q", prefix)[0]
            if count != expected:
                raise ValueError("tensor %d length mismatch: %d != %d" %
                                 (index, count, expected))
            payload = source.read(count * 4)
            if len(payload) != count * 4:
                raise ValueError("truncated tensor payload at index %d" % index)
        if source.read(1):
            raise ValueError("trailing bytes after final tensor")
    return config


def stage_model(source_path):
    fd, temporary = tempfile.mkstemp(prefix=".promotion-", suffix=".net",
                                     dir=str(PUBLIC_DIR))
    digest = hashlib.sha256()
    size = 0
    try:
        with source_path.open("rb") as source, os.fdopen(fd, "wb") as output:
            before = os.fstat(source.fileno())
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            after = os.fstat(source.fileno())
            output.flush()
            os.fsync(output.fileno())
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("source model changed while being staged")
        staged = Path(temporary)
        config = validate_model(staged)
        return staged, config, digest.hexdigest(), size
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def fetch_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "az-promote/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def fetch_sha(url):
    request = urllib.request.Request(url, headers={"User-Agent": "az-promote/1.0"})
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(request, timeout=60) as response:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def verify_online(manifest, digest, size):
    channel_url = PUBLIC_BASE + "/channels/stable.json?cb=%d" % int(time.time())
    last_error = None
    # Static Asset propagation can briefly return the previous generation or
    # a 404 immediately after Wrangler reports success. Retry the readback;
    # promotion is complete only after both pointer and immutable asset match.
    for attempt in range(12):
        try:
            online = fetch_json(channel_url + "-%d" % attempt)
            if online != manifest:
                raise ValueError("stable channel still contains older metadata")
            online_sha, online_size = fetch_sha(online["file"])
            if online_sha != digest or online_size != size:
                raise ValueError("immutable asset hash/size mismatch")
            return
        except Exception as error:  # network/HTTP/propagation mismatch
            last_error = error
            if attempt + 1 < 12:
                time.sleep(min(2 + attempt, 10))
    raise ValueError("online verification failed after retries: %s" % last_error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--deploy", action="store_true")
    args = parser.parse_args()

    model = args.model.resolve()
    if not model.is_file():
        raise SystemExit("model not found: %s" % model)
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.label):
        raise SystemExit("label must match [a-z0-9][a-z0-9-]*")

    previous_manifest = json.loads(CHANNEL_FILE.read_text(encoding="utf-8"))
    staged, header, digest, staged_size = stage_model(model)
    manifest = dict(previous_manifest)
    expected = manifest["config"]
    for key in (
        "input_channels", "board_height", "board_width", "trunk_channels",
        "residual_blocks", "policy_channels", "policy_size",
        "value_channels", "value_hidden_dim",
    ):
        if header[key] != expected[key]:
            raise SystemExit("model config mismatch for %s: %s != %s" %
                             (key, header[key], expected[key]))

    asset_name = "%s-%s.net" % (args.label, digest[:8])
    destination = PUBLIC_DIR / asset_name
    if destination.exists():
        existing = hashlib.sha256(destination.read_bytes()).hexdigest()
        if existing != digest:
            raise SystemExit("refusing to replace mismatched immutable asset")
        staged.unlink()
    else:
        os.replace(str(staged), str(destination))

    manifest.update({
        "channel": "stable",
        "release": args.release,
        "file": "%s/%s" % (PUBLIC_BASE, asset_name),
        "sha256": digest,
        "size_bytes": staged_size,
        "variant": args.variant,
    })
    atomic_json(CHANNEL_FILE, manifest)

    print("asset:", destination)
    print("stable:", CHANNEL_FILE)
    print("sha256:", digest)

    if not args.deploy:
        return
    try:
        subprocess.run(["npx", "wrangler", "deploy"], cwd=MODEL_DIR / "worker",
                       check=True)
        verify_online(manifest, digest, staged_size)
    except Exception:
        print("deployment verification failed; rolling stable channel back",
              file=sys.stderr)
        atomic_json(CHANNEL_FILE, previous_manifest)
        subprocess.run(["npx", "wrangler", "deploy"], cwd=MODEL_DIR / "worker",
                       check=True)
        verify_online(previous_manifest, previous_manifest["sha256"],
                      previous_manifest["size_bytes"])
        raise
    print("online verification: OK")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print("promote failed:", error, file=sys.stderr)
        sys.exit(1)
