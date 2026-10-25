"""Isolate persistence to a temp dir for the whole test session."""
import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="duat-test-")
os.environ["DUAT_DB_PATH"] = os.path.join(_TMP, "duat.db")
os.environ["DUAT_ARTIFACT_DIR"] = os.path.join(_TMP, "artifacts", "sha256")
