"""The local server must refuse other host names and cross-site requests."""
import io
import unittest

import tests.conftest_path  # noqa: F401
from main import request_blocked_reason
from tests.test_app import NOTES, make_client


class GuardLogic(unittest.TestCase):
    def test_hosts(self):
        for host in ("127.0.0.1:5000", "localhost:61234", "localhost", "[::1]:5000", "LOCALHOST:5000"):
            self.assertEqual(request_blocked_reason("GET", host, None, None), "", host)
        for host in ("evil.example", "evil.example:5000", "localhost.evil.example:5000",
                     "127.0.0.1.nip.io:5000", ""):
            self.assertEqual(request_blocked_reason("GET", host, None, None), "host", host)

    def test_state_changing_requests(self):
        h = "127.0.0.1:5000"
        self.assertEqual(request_blocked_reason("POST", h, "http://127.0.0.1:5000", "same-origin"), "")
        self.assertEqual(request_blocked_reason("POST", h, None, None), "")          # non-browser client
        self.assertEqual(request_blocked_reason("POST", h, None, "none"), "")
        self.assertEqual(request_blocked_reason("POST", h, "https://evil.example", None), "origin")
        self.assertEqual(request_blocked_reason("POST", h, "null", None), "origin")
        self.assertEqual(request_blocked_reason("POST", h, "http://127.0.0.1:8080", None), "origin")
        self.assertEqual(request_blocked_reason("POST", h, None, "cross-site"), "cross-site")
        self.assertEqual(request_blocked_reason("POST", h, None, "same-site"), "cross-site")
        # reads are left to the browser's same-origin policy
        self.assertEqual(request_blocked_reason("GET", h, "https://evil.example", "cross-site"), "")


class GuardOnApi(unittest.TestCase):
    def test_dns_rebinding_host_cannot_read_status(self):
        c = make_client()
        r = c.get("/api/status", base_url="http://attacker.example:5000/")
        self.assertEqual(r.status_code, 403)
        self.assertIn("error", r.get_json())

    def test_other_site_cannot_upload_or_clear(self):
        c = make_client()
        r = c.post("/api/upload", data={"file": (io.BytesIO(NOTES), "a.txt")},
                   content_type="multipart/form-data", headers={"Origin": "https://evil.example"})
        self.assertEqual(r.status_code, 403)
        r = c.post("/api/clear", headers={"Sec-Fetch-Site": "cross-site"})
        self.assertEqual(r.status_code, 403)
        r = c.post("/api/sample", json={}, headers={"Origin": "http://localhost:9999"})
        self.assertEqual(r.status_code, 403)

    def test_own_page_is_allowed(self):
        c = make_client()
        r = c.post("/api/clear", base_url="http://127.0.0.1:5000/",
                   headers={"Origin": "http://127.0.0.1:5000", "Sec-Fetch-Site": "same-origin"})
        self.assertEqual(r.status_code, 200)

    def test_security_headers(self):
        r = make_client().get("/")
        self.assertEqual(r.headers["X-Content-Type-Options"], "nosniff")
        self.assertIn("frame-ancestors 'none'", r.headers["Content-Security-Policy"])


if __name__ == "__main__":
    unittest.main()
