import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
W = '/home/user/Six-Nations-Fantasy/.claude/worktrees/agent-a632f81a96abfb8ca'
S = '/tmp/claude-0/-home-user-Six-Nations-Fantasy/f11ffa4a-ba0b-5432-952a-6accc17d3a46/scratchpad'
B = S + '/agentB'
if W not in sys.path:
    sys.path.insert(0, W)
os.chdir(W)
