import tempfile
import unittest
from pathlib import Path
from scripts.gate2_m_api_smoke import credentials, NoRedirect


class CredentialBoundaryTests(unittest.TestCase):
    def test_reject_ambiguous_or_credential_bearing_urls(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "private.txt"
            for content in (
                "sk-test\nsk-second\nhttps://example.com/v1",
                "sk-test\nhttps://user:password@example.com/v1",
                "sk-test\nhttps://example.com/v1?key=secret",
                "sk-test\nhttp://example.com/v1",
            ):
                path.write_text(content)
                with self.subTest(content=content), self.assertRaises(ValueError):
                    credentials(path)

    def test_redirect_is_not_followed(self):
        self.assertIsNone(NoRedirect().redirect_request(None, None, 302, "", {}, "https://other.test"))
