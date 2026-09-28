# Manner physical-metric baseline

Trajectories: 60 | groups: 7


## Per-group summary (mean +/- std)

| task/tag | n | SPARC | LDLJ | peak_jspeed | app/desc ratio | max/min ratio | peak_grasp_grip | mean_grasp_grip |
|---|---|---|---|---|---|---|---|---|
| lift/phased_5stage | 7 | -4.244+/-0.334 | -17.902+/-0.212 | 1.588+/-0.002 | 3.517+/-0.268 | 295.628+/-61.588 | 0.000+/-0.000 | 0.000+/-0.000 |
| pour/fast | 10 | -4.586+/-0.394 | -22.035+/-0.069 | 1.620+/-0.007 | 0.553+/-0.037 | 323.677+/-23.404 | 0.000+/-0.000 | 0.000+/-0.000 |
| pour/gentle | 10 | -6.361+/-0.305 | -21.859+/-0.076 | 1.579+/-0.001 | 0.478+/-0.030 | 249.839+/-15.763 | 0.000+/-0.000 | 0.000+/-0.000 |
| pour/phased_5stage | 6 | -15.285+/-0.478 | -25.207+/-0.112 | 1.584+/-0.001 | 4.945+/-0.157 | 816.141+/-234.613 | 0.000+/-0.000 | 0.000+/-0.000 |
| stack/firm_grip | 10 | -6.530+/-0.509 | -20.737+/-0.454 | 1.593+/-0.012 | 0.477+/-0.083 | 161.841+/-72.433 | 35.170+/-9.914 | 27.878+/-9.082 |
| stack/gentle_grip | 10 | -7.411+/-1.083 | -23.255+/-0.465 | 1.599+/-0.010 | 0.578+/-0.111 | 98.452+/-41.887 | 0.000+/-0.000 | 0.000+/-0.000 |
| stack/phased_5stage | 7 | -17.136+/-5.917 | -22.758+/-1.921 | 1.588+/-0.002 | 8.076+/-2.031 | 4764.616+/-5430.493 | 0.000+/-0.000 | 0.000+/-0.000 |

## Cohen's d: fast-family vs gentle-family

|d| > 0.8 = large effect (manner physically separable).


### task = pour

| field | Cohen's d |
|---|---|
| sparc | 5.036 |
| ldlj | -2.434 |
| sparc_joint | 28.376 |
| eef_path_length | -0.782 |
| integral_kinetic_energy | 14.262 |
| peak_cfrc | 0.000 |
| mean_cfrc | 0.000 |
| contact_approach_velocity | 0.000 |
| post_contact_force_slope | 0.000 |
| mean_joint_speed | 257.095 |
| peak_joint_speed | 8.397 |
| peak_grasp_force | 0.000 |
| mean_grasp_force | 0.000 |
| peak_grasp_force_grip | 0.000 |
| mean_grasp_force_grip | 0.000 |

## Cohen's d: firm-grip vs gentle-grip (contact-force axis)

True finger-object normal contact force via mj_contactForce; |d| > 0.8 = large effect.


### task = stack

| field | Cohen's d |
|---|---|
| sparc | 1.041 |
| ldlj | 5.481 |
| sparc_joint | 0.006 |
| eef_path_length | -0.808 |
| integral_kinetic_energy | -0.708 |
| peak_cfrc | -0.665 |
| mean_cfrc | -0.665 |
| contact_approach_velocity | -0.145 |
| post_contact_force_slope | -0.070 |
| mean_joint_speed | 0.975 |
| peak_joint_speed | -0.510 |
| peak_grasp_force | 5.024 |
| mean_grasp_force | 3.576 |
| peak_grasp_force_grip | 5.017 |
| mean_grasp_force_grip | 4.341 |

## Manner-classifier leave-one-out AUC

Out-of-sample AUC of a diagonal-LDA manner classifier (LOO cross-validated). 1.0 = perfectly separable, 0.5 = chance.


### Kinematic axis (fast vs gentle)

| task | n(a/b) | multivariate LOO AUC | best single-feature AUC |
|---|---|---|---|
| pour | 10/10 | 1.000 | sparc=1.000 |

### Contact-force axis (firm-grip vs gentle-grip)

| task | n(a/b) | multivariate LOO AUC | best single-feature AUC |
|---|---|---|---|
| stack | 10/10 | 0.900 | peak_grasp_force=1.000 |

## Trajectory-shape separability (silhouette + DTW/Frechet)

Silhouette on start-centred eef-path signatures: +1 = tight, well-separated manner clusters, ~0 = overlapping. DTW/Frechet are distances between the two groups' mean paths (larger = more physically distinct shapes).


### Kinematic axis (fast vs gentle)

| task | n(a/b) | silhouette | DTW(mean paths) | Frechet(mean paths) |
|---|---|---|---|---|
| pour | 10/10 | 0.634 | 1.363 | 0.056 |

### Contact-force axis (firm-grip vs gentle-grip)

| task | n(a/b) | silhouette | DTW(mean paths) | Frechet(mean paths) |
|---|---|---|---|---|
| stack | 10/10 | 0.029 | 3.770 | 0.066 |
