"""Verify a freshly built DLL without attaching to the game. Pass DLL path as argv[1]."""
import ctypes
import json
import threading
from pathlib import Path
from pvz_gym.bridge.native import NativePvzBackend, _call_transition_with_resizable_buffer

import sys
path = Path(sys.argv[1]).resolve()
backend=NativePvzBackend(path)
d=backend._dll
d.pvz_gym_env_step_headless.argtypes=[ctypes.c_int]*8+[ctypes.c_char_p,ctypes.c_int]
d.pvz_gym_env_step_headless.restype=ctypes.c_int
d.pvz_gym_set_step_profiling.argtypes=[ctypes.c_bool]
d.pvz_gym_get_step_timing_json.argtypes=[ctypes.c_char_p,ctypes.c_int]
d.pvz_gym_get_step_timing_json.restype=ctypes.c_int
d.pvz_gym_set_step_profiling(True)
request=d.pvz_gym_begin_result_request()
small=ctypes.create_string_buffer(2)
ret=d.pvz_gym_env_step_headless(*([0]*8),small,2)
assert ret < -2
assert d.pvz_gym_read_result_request(request+1,small,2)==-1
foreign=[]
def check_thread():
    foreign.append(d.pvz_gym_read_result_request(request,small,2))
t=threading.Thread(target=check_thread);t.start();t.join()
assert foreign==[-1]
large=ctypes.create_string_buffer(-ret)
assert d.pvz_gym_read_result_request(request,large,-ret)>=0
result=json.loads(large.value)
assert result['message']=='ticks must be positive'
assert d.pvz_gym_read_result_request(request,large,-ret)==-1
timing=ctypes.create_string_buffer(2048)
assert d.pvz_gym_get_step_timing_json(timing,2048)>=0
assert json.loads(timing.value)['serialization']>=0
executions=[]
def execute(buf,size):
    executions.append(1)
    return d.pvz_gym_env_step_headless(*([0]*8),buf,size)
ret,payload=backend._call_transition(execute,initial_size=2)
assert len(executions)==1 and json.loads(payload)==result
print('Built DLL ABI: request identity, thread isolation, consume-once, opt-in timing and one-execution resize passed; no game attachment.')
