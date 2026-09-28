# Manner physical-metric baseline

Trajectories: 57 | groups: 15


## Per-group summary (mean +/- std)

| task/tag | n | SPARC | LDLJ | peak_jspeed | app/desc ratio | max/min ratio | peak_grasp_grip | mean_grasp_grip |
|---|---|---|---|---|---|---|---|---|
| lift/phased_5stage | 7 | -4.244+/-0.334 | -17.902+/-0.212 | 1.588+/-0.002 | 3.517+/-0.268 | 295.628+/-61.588 | 0.000+/-0.000 | 0.000+/-0.000 |
| pour/baseline | 2 | -5.047+/-0.404 | -21.048+/-0.045 | 1.658+/-0.061 | 0.772+/-0.068 | 167.695+/-9.002 | 0.000+/-0.000 | 0.000+/-0.000 |
| pour/fast | 2 | -3.866+/-0.086 | -21.976+/-0.110 | 1.796+/-0.174 | 0.965+/-0.008 | 325.349+/-10.498 | 0.000+/-0.000 | 0.000+/-0.000 |
| pour/fast_en | 2 | -4.021+/-0.398 | -21.995+/-0.263 | 1.789+/-0.173 | 0.907+/-0.120 | 305.956+/-38.937 | 0.000+/-0.000 | 0.000+/-0.000 |
| pour/fast_then_slow | 2 | -6.192+/-1.040 | -18.931+/-0.806 | 1.584+/-0.003 | 3.444+/-3.706 | 194.895+/-89.225 | 0.000+/-0.000 | 0.000+/-0.000 |
| pour/firm | 2 | -4.791+/-0.033 | -21.213+/-0.221 | 1.655+/-0.070 | 0.721+/-0.002 | 266.058+/-6.040 | 0.000+/-0.000 | 0.000+/-0.000 |
| pour/gentle | 2 | -5.453+/-0.094 | -22.132+/-0.633 | 1.579+/-0.000 | 0.735+/-0.106 | 229.017+/-39.691 | 0.000+/-0.000 | 0.000+/-0.000 |
| pour/gentle_en | 2 | -6.695+/-0.340 | -24.572+/-0.824 | 1.600+/-0.019 | 0.708+/-0.104 | 45.175+/-3.148 | 0.000+/-0.000 | 0.000+/-0.000 |
| pour/normal | 2 | -5.004+/-0.289 | -21.088+/-0.046 | 1.656+/-0.072 | 0.711+/-0.043 | 162.816+/-9.411 | 0.000+/-0.000 | 0.000+/-0.000 |
| pour/phased_5stage | 7 | -15.290+/-0.437 | -25.163+/-0.155 | 1.584+/-0.002 | 5.299+/-0.949 | 1938.276+/-2976.606 | 0.000+/-0.000 | 0.000+/-0.000 |
| pour/ultra_careful | 2 | -5.110+/-0.148 | -23.176+/-0.677 | 1.580+/-0.001 | 0.424+/-0.069 | 221.197+/-16.557 | 0.000+/-0.000 | 0.000+/-0.000 |
| pour/ultra_gentle | 2 | -5.551+/-0.442 | -23.100+/-0.753 | 1.580+/-0.000 | 0.394+/-0.048 | 201.258+/-10.742 | 0.000+/-0.000 | 0.000+/-0.000 |
| stack/firm_grip | 8 | -6.752+/-0.958 | -21.018+/-0.542 | 1.594+/-0.011 | 0.488+/-0.072 | 167.606+/-66.323 | 34.494+/-11.072 | 28.074+/-10.618 |
| stack/gentle_grip | 8 | -7.426+/-0.960 | -23.207+/-0.495 | 1.597+/-0.006 | 0.577+/-0.120 | 92.555+/-38.803 | 0.000+/-0.000 | 0.000+/-0.000 |
| stack/phased_5stage | 7 | -17.136+/-5.917 | -22.758+/-1.921 | 1.588+/-0.002 | 8.076+/-2.031 | 4764.616+/-5430.493 | 0.000+/-0.000 | 0.000+/-0.000 |

## Cohen's d: fast-family vs gentle-family

|d| > 0.8 = large effect (manner physically separable).


### task = pour

| field | Cohen's d |
|---|---|
| sparc | 1.024 |
| ldlj | 1.936 |
| sparc_joint | 1.499 |
| eef_path_length | -0.232 |
| integral_kinetic_energy | 2.440 |
| peak_cfrc | 0.000 |
| mean_cfrc | 0.000 |
| contact_approach_velocity | 0.000 |
| post_contact_force_slope | 0.000 |
| mean_joint_speed | 2.501 |
| peak_joint_speed | 1.199 |
| peak_grasp_force | 0.000 |
| mean_grasp_force | 0.000 |
| peak_grasp_force_grip | 0.000 |
| mean_grasp_force_grip | 0.000 |

## Cohen's d: firm-grip vs gentle-grip (contact-force axis)

True finger-object normal contact force via mj_contactForce; |d| > 0.8 = large effect.


### task = stack

| field | Cohen's d |
|---|---|
| sparc | 0.703 |
| ldlj | 4.217 |
| sparc_joint | 0.209 |
| eef_path_length | -0.200 |
| integral_kinetic_energy | -0.038 |
| peak_cfrc | -1.852 |
| mean_cfrc | -1.852 |
| contact_approach_velocity | -0.574 |
| post_contact_force_slope | 0.026 |
| mean_joint_speed | 0.317 |
| peak_joint_speed | -0.348 |
| peak_grasp_force | 4.404 |
| mean_grasp_force | 3.235 |
| peak_grasp_force_grip | 4.406 |
| mean_grasp_force_grip | 3.739 |
