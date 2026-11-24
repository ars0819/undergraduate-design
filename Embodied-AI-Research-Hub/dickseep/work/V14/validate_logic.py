import ast, contextlib, hashlib, io, json, pathlib, platform, sys
from collections import deque
from types import SimpleNamespace
root = pathlib.Path('work/V14')
source = root / 'source/experiments/robot/libero/run_libero_eval.py'
original = source.read_text(encoding='utf-8')
results = {'python': sys.version, 'platform': platform.platform(), 'seed': None, 'seed_reason': 'deterministic symbolic indices; no RNG', 'P': 8, 'cases': []}
for E in (1, 2, 4, 8):
    q = deque(maxlen=E)
    q.extend(range(8))
    after = list(q)
    popped = [q.popleft() for _ in range(len(q))]
    assert after == popped == list(range(8-E, 8))
    fixed = deque(maxlen=E)
    fixed.extend(list(range(8))[:E])
    assert list(fixed) == list(range(E))
    for patched in (False, True):
        tree = ast.parse(original)
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_episode')
        if patched:
            replacements = 0
            for n in ast.walk(node):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'extend' and isinstance(n.func.value, ast.Name) and n.func.value.id == 'action_queue':
                    n.args[0] = ast.parse('actions[:cfg.num_open_loop_steps]', mode='eval').body
                    replacements += 1
            assert replacements == 1
        events, executed = [], []
        class Action:
            def __init__(self, block, index): self.block, self.index = block, index
            def tolist(self): return [self.block, self.index]
        class Env:
            def reset(self): pass
            def get_observation(self): return {}
            def step(self, action):
                events.append(['step', *action]); executed.append(action)
                return {}, 0, False, {}
        calls = [0]
        def get_action(*args, **kwargs):
            block = calls[0]; calls[0] += 1
            events.append(['infer_start', block])
            actions = [Action(block, i) for i in range(8)]
            events.append(['infer_end', block])
            return actions
        errors = []
        scope = {'GenerateConfig': object, 'deque': deque, 'NUM_ACTIONS_CHUNK': 8, 'TASK_MAX_STEPS': {'test': E*2},
                 'get_action': get_action, 'prepare_observation': lambda obs, size: ({}, None),
                 'process_action': lambda action, family: action, 'log_message': lambda *args: errors.append(str(args))}
        module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
        exec(compile(module, str(source), 'exec'), scope)
        cfg = SimpleNamespace(num_open_loop_steps=E, num_steps_wait=0, task_suite_name='test', model_family='openvla', use_film=False)
        with contextlib.redirect_stdout(io.StringIO()):
            scope['run_episode'](cfg, Env(), 'symbolic test', None, None)
        expected_indices = list(range(E)) if patched else list(range(8-E,8))
        expected = [[b,i] for b in range(2) for i in expected_indices]
        assert executed == expected, (executed, expected)
        assert calls[0] == 2 and not errors
        assert events == sum(([['infer_start',b],['infer_end',b]] + [['step',b,i] for i in expected_indices] for b in range(2)), [])
        results['cases'].append({'E': E, 'in_memory_prefix_fix': patched, 'queue_after_extend_original': after, 'executed': executed, 'events': events, 'assertions': 'PASS'})
results['result'] = 'PASS: 4 deque traces and 8 AST-extracted run_episode tests (two chunks each)'
(root/'logic_validation.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
print(json.dumps(results, indent=2))

