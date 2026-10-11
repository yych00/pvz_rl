"""Private release access and credential isolation during asset redirects."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

from scripts.download_native import download_native, SafeRedirectHandler, ReleaseDownloadError


class ReleaseDownloadTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.target = Path(self.temporary.name)
        self.filename = 'pvz-rl-native-0.1.0-windows-x64.zip'
        self.archive = b'archive fixture'
        self.metadata = json.dumps({'assets': [{'name': self.filename, 'id': 1},
                                  {'name': self.filename + '.sha256', 'id': 2}]}).encode()

    def asset(self, path, token, *, asset=False):
        if path.endswith('/tags/v0.1.0'):
            return self.metadata
        if path.endswith('/assets/1'):
            return self.archive
        return (hashlib.sha256(self.archive).hexdigest() + '  ' + self.filename).encode()

    @patch('scripts.download_native.version', return_value='0.1.0')
    def test_public_release_needs_no_credentials(self, _version):
        with patch.dict(os.environ, {'GH_TOKEN': '', 'GITHUB_TOKEN': ''}), \
             patch('scripts.download_native._read_api', side_effect=self.asset) as read, \
             patch('scripts.download_native._git_token') as credential:
            path = download_native('owner/project', self.target)
        self.assertEqual(path.read_bytes(), self.archive)
        credential.assert_not_called()
        self.assertTrue(all(call.args[1] is None for call in read.call_args_list))

    @patch('scripts.download_native.version', return_value='0.1.0')
    def test_private_release_reuses_git_login(self, _version):
        def private(path, token, **kwargs):
            if token is None:
                raise urllib.error.HTTPError('https://api.github.com', 404, 'private', {}, None)
            return self.asset(path, token, **kwargs)
        with patch.dict(os.environ, {'GH_TOKEN': '', 'GITHUB_TOKEN': ''}), \
             patch('scripts.download_native._git_token', return_value='fixture-secret'), \
             patch('scripts.download_native._read_api', side_effect=private) as read:
            download_native('owner/project', self.target)
        self.assertEqual(read.call_count, 4)
        self.assertTrue(all(call.args[1] == 'fixture-secret' for call in read.call_args_list[1:]))

    def test_credentials_removed_on_cdn_redirect(self):
        request = urllib.request.Request('https://api.github.com/repos/owner/project/releases/assets/1',
                                         headers={'Authorization': 'Bearer fixture-secret'})
        redirected = SafeRedirectHandler().redirect_request(request, None, 302, 'redirect', {},
                                                            'https://release-assets.githubusercontent.com/asset')
        self.assertFalse(redirected.has_header('Authorization'))
        with self.assertRaises(ReleaseDownloadError):
            SafeRedirectHandler().redirect_request(request, None, 302, 'redirect', {}, 'http://example.com/asset')

    @patch('scripts.download_native.version', return_value='0.1.0')
    def test_bad_checksum_preserves_existing_archive(self, _version):
        existing = self.target / self.filename
        existing.write_bytes(b'keep')
        def corrupt(path, token, **kwargs):
            return b'0' * 64 if path.endswith('/assets/2') else self.asset(path, token, **kwargs)
        with patch('scripts.download_native._read_api', side_effect=corrupt), self.assertRaises(ReleaseDownloadError):
            download_native('owner/project', self.target)
        self.assertEqual(existing.read_bytes(), b'keep')
        self.assertEqual(list(self.target.glob('*.incoming')), [])

    @patch('scripts.download_native.version', return_value='0.1.0')
    def test_empty_checksum_does_not_write_archive(self, _version):
        def empty_checksum(path, token, **kwargs):
            return b'' if path.endswith('/assets/2') else self.asset(path, token, **kwargs)
        with patch('scripts.download_native._read_api', side_effect=empty_checksum), self.assertRaises(ReleaseDownloadError):
            download_native('owner/project', self.target)
        self.assertFalse((self.target / self.filename).exists())


if __name__ == '__main__':
    unittest.main()
