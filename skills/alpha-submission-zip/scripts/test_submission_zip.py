import importlib.util
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

spec = importlib.util.spec_from_file_location("submission_zip", Path(__file__).with_name("submission_zip.py"))
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


class ArchiveTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "solution.py").write_text("from agent import run\nrun()\n")
        (self.root / "agent.py").write_text("def run():\n    return 1\n")
        self.baseline = self.root / "baseline.py"
        self.baseline.write_text("# baseline interface fixture\n")
        self.requirements = self.root / "organizer_requirements.txt"
        self.requirements.write_bytes(b"# Python 3.12\nnumpy==1.26.4\n")
        (self.root / "requirements.txt").write_bytes(self.requirements.read_bytes())
        self.output = self.root / "dist"
        self.files = ["solution.py", "agent.py", "requirements.txt"]

    def tearDown(self):
        self.temporary.cleanup()

    def save(self):
        return package.build(self.root, self.files, self.baseline, self.output, self.requirements)

    def test_versions_increment_without_changing_previous_archive(self):
        first = self.save()
        original = Path(first["archive_path"]).read_bytes()
        (self.root / "agent.py").write_text("def run():\n    return 2\n")
        second = self.save()
        self.assertEqual(first["version"], 1)
        self.assertEqual(second["version"], 2)
        self.assertEqual(Path(second["archive_path"]).name, "solution_v002.zip")
        self.assertEqual(Path(first["archive_path"]).read_bytes(), original)
        self.assertNotEqual(first["sha256"], second["sha256"])

    def test_gaps_and_manifest_only_reservations_are_not_reused(self):
        self.save()
        (self.output / "solution_v009.manifest.json").write_text("{}")
        self.assertEqual(self.save()["version"], 10)

    def test_broken_source_does_not_publish_or_replace_archive(self):
        first = self.save()
        (self.root / "agent.py").write_text("def broken(:\n")
        with self.assertRaises(SyntaxError):
            self.save()
        self.assertFalse((self.output / "solution_v002.zip").exists())
        self.assertEqual(package.inspect_archive(first["archive_path"], requirements=self.requirements)["sha256"], first["sha256"])

    def test_parseable_but_uncompilable_source_is_rejected(self):
        (self.root / "agent.py").write_text("return 1\n")
        with self.assertRaises(SyntaxError):
            self.save()
        self.assertFalse((self.output / "solution_v001.zip").exists())

    def test_validation_requires_the_target_python_version(self):
        for version in ((3, 11, 9), (3, 13, 0)):
            with self.subTest(version=version), patch.object(sys, "version_info", version):
                with self.assertRaisesRegex(ValueError, "Python 3.12"):
                    self.save()
        self.assertFalse((self.output / "solution_v001.zip").exists())

    def test_manifest_does_not_claim_runtime_or_embedded_dependency_checks(self):
        report = self.save()
        self.assertTrue(report["python_version"].startswith("3.12."))
        self.assertFalse(report["runtime_interface_checked"])
        self.assertFalse(report["dependency_policy_checked"])
        self.assertFalse(report["embedded_payload_checked"])

    def test_missing_entrypoint_secrets_and_path_traversal_are_rejected(self):
        for names in (["agent.py", "requirements.txt"], ["solution.py"], *[self.files + [name] for name in (".env", "../outside.py", "/tmp/absolute.py", "train.csv")]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                package.build(self.root, names, self.baseline, self.output, self.requirements)

    def test_wrapper_folder_and_duplicate_members_are_rejected(self):
        path = self.root / "bad.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("wrapper/solution.py", "pass")
        with self.assertRaisesRegex(ValueError, "ZIP root"):
            package.inspect_archive(path, requirements=self.requirements)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("solution.py", "pass")
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                archive.writestr("solution.py", "pass")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            package.inspect_archive(path, requirements=self.requirements)

    def test_symlink_member_and_source_mismatch_are_rejected(self):
        path = self.root / "bad.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("solution.py", "pass")
            archive.writestr("requirements.txt", self.requirements.read_bytes())
            link = zipfile.ZipInfo("link.py")
            link.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(link, "/outside")
        with self.assertRaisesRegex(ValueError, "symlink"):
            package.inspect_archive(path, requirements=self.requirements)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("solution.py", "pass")
            archive.writestr("requirements.txt", self.requirements.read_bytes())
            directory = zipfile.ZipInfo("pretend.py")
            directory.external_attr = (stat.S_IFDIR | 0o755) << 16
            archive.writestr(directory, "")
        with self.assertRaisesRegex(ValueError, "directory"):
            package.inspect_archive(path, requirements=self.requirements)
        first = self.save()
        (self.root / "agent.py").write_text("# changed after packaging")
        with self.assertRaisesRegex(ValueError, "differs from source"):
            package.inspect_archive(first["archive_path"], self.root, self.requirements)

    def test_nested_modules_directories_and_non_python_assets_are_rejected(self):
        path = self.root / "bad.zip"
        for name in ("agent/controller.py", "empty/", "knowledge.json", "guide.md", "index.sqlite", "package.whl", ".env.example", "config.txt"):
            with self.subTest(name=name):
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr("solution.py", "pass")
                    archive.writestr("requirements.txt", self.requirements.read_bytes())
                    archive.writestr(name, "")
                with self.assertRaises(ValueError):
                    package.inspect_archive(path, requirements=self.requirements)

    def test_missing_or_changed_requirements_are_rejected(self):
        path = self.root / "bad.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("solution.py", "pass")
        with self.assertRaisesRegex(ValueError, "requirements.txt"):
            package.inspect_archive(path, requirements=self.requirements)
        (self.root / "requirements.txt").write_text("numpy==2.0.0\n")
        with self.assertRaisesRegex(ValueError, "differs from the original"):
            self.save()
        self.assertFalse((self.output / "solution_v001.zip").exists())

    def test_oversized_archive_is_rejected_before_publishing(self):
        first = self.save()
        size = Path(first["archive_path"]).stat().st_size
        with patch.object(package, "MAX_ARCHIVE_BYTES", size):
            package.inspect_archive(first["archive_path"], requirements=self.requirements)
        with patch.object(package, "MAX_ARCHIVE_BYTES", size - 1):
            with self.assertRaisesRegex(ValueError, "100 MB"):
                package.inspect_archive(first["archive_path"], requirements=self.requirements)
            with self.assertRaisesRegex(ValueError, "100 MB"):
                self.save()
        self.assertFalse((self.output / "solution_v002.zip").exists())
        self.assertEqual(package.inspect_archive(first["archive_path"], requirements=self.requirements)["sha256"], first["sha256"])

    def test_platform_copy_runs_using_only_root_python_files_and_requirements(self):
        (self.root / "agent.py").write_text("def run():\n    from pathlib import Path\n    Path('submission.csv').write_text('id,prediction\\n1,0\\n')\n")
        report = self.save()
        task_directory = self.root / "clean_task"
        task_directory.mkdir()
        with zipfile.ZipFile(report["archive_path"]) as archive:
            for name in archive.namelist():
                self.assertNotIn("/", name)
                self.assertTrue(name.endswith(".py") or name == "requirements.txt")
                (task_directory / name).write_bytes(archive.read(name))
        (task_directory / "task.md").write_text("Technical fixture")
        result = subprocess.run([sys.executable, "solution.py", "task.md"], cwd=task_directory,
            env={"PATH": os.defpath, "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"},
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((task_directory / "submission.csv").read_text(), "id,prediction\n1,0\n")


if __name__ == "__main__":
    unittest.main()
