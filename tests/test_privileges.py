import unittest
from unittest.mock import Mock, patch

from live_gpt.privileges import foreground_requires_administrator


class PrivilegeTests(unittest.TestCase):
    def probe(self, elevations, *, token_ok=True):
        kernel, advapi, user = Mock(), Mock(), Mock()
        kernel.OpenProcess.side_effect = lambda _access, _inherit, pid: pid
        user.GetForegroundWindow.return_value = 100

        def foreground(_hwnd, pid):
            pid._obj.value = 2
            return 1

        def token(pid, _access, out):
            out._obj.value = pid + 10 if token_ok else 0
            return token_ok

        def information(handle, _kind, value, _size, _returned):
            elevation = elevations[handle.value - 10]
            if elevation is None:
                return False
            value._obj.value = elevation
            return True

        user.GetWindowThreadProcessId.side_effect = foreground
        advapi.OpenProcessToken.side_effect = token
        advapi.GetTokenInformation.side_effect = information
        with patch('live_gpt.privileges.sys.platform', 'win32'), \
             patch('live_gpt.privileges.os.getpid', return_value=1), \
             patch('live_gpt.privileges.ctypes.WinDLL', create=True,
                   side_effect=[kernel, advapi, user]):
            result = foreground_requires_administrator()
        return result, kernel

    def test_mismatch_and_handle_cleanup(self):
        result, kernel = self.probe({1: False, 2: True})
        self.assertTrue(result)
        self.assertEqual(kernel.CloseHandle.call_count, 4)

    def test_no_warning_for_equal_privileges_or_unknown(self):
        for elevations in ({1: True}, {1: False, 2: False},
                           {1: None}, {1: False, 2: None}):
            with self.subTest(elevations=elevations):
                self.assertFalse(self.probe(elevations)[0])

    def test_token_access_failure_closes_process(self):
        result, kernel = self.probe({}, token_ok=False)
        self.assertFalse(result)
        kernel.CloseHandle.assert_called_once_with(1)
