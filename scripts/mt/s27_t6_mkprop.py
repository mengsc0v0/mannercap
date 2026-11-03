import json, sys
R, A, D, O = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
p = json.load(open('reports/step26_llm_onboard/t6/v1/llm_loop/round5_proposal.json'))
pm = p['proposal']['objects'][1]['params']; pm.pop('grasp_candidate', None)
pm['grasp_wall_pinch'] = {'panel_angle_deg': A, 'depth_m': D}
p['step27'] = {'base': 't6/v1/llm_loop/round5_proposal.json', 'change': 'mug: grasp_candidate -> grasp_wall_pinch', 'round': R}
if R >= 2:
    p['proposal']['objects'][0]['params']['grasp_grid'] = {'yaws_deg': [0, 180], 'laterals_m': [0.0, 0.01, -0.01, 0.02, -0.02], 'dzs_m': [0.008]}
    p['step27']['pudding'] = 'existing grasp_grid, dz=0.008: fingertips 13.2 mm below site vs origin 13.1 mm above table -> 8 mm clearance'
json.dump(p, open(O + '/proposal.json', 'w'), indent=1)
