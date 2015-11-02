
beam=10
retry_beam=40

mono_num_iters = 40

scale_opts = ['--transition-scale=1.0',
                '--acoustic-scale=0.1',
                '--self-loop-scale=0.1']

mono_max_iter_inc = 30
totgauss = 1000
boost_silence = 1.0
mono_realign_iters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 23, 26, 29, 32, 35, 38]
stage = -4
power = 0.25

tri_num_iters = 35

tri_max_iter_inc = 25

tri_realign_iters = [10, 20, 30]

tri_num_states = 3100
tri_num_gauss = 50000

fmllr_update_type = 'full'
fmllr_iters=[2, 4, 6, 12]
silence_weight=0.0
fmllr_power = 0.2
