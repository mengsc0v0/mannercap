# Third-party source versions

This file records the source revisions bundled by
`stagecap_robosuite_libero_repro_20260714.tar.gz`.

| Component | Upstream | Revision | Local changes |
| --- | --- | --- | --- |
| RoboSuite 1.5.1 | `https://github.com/uynitsuj/robosuite` | `97292732ed909ac3ae116579fb768607034a4dbd` | none |
| LIBERO-PRO 0.1.0 | `https://github.com/uynitsuj/LIBERO-PRO.git` | `5368540790fde9a18584d64105ec5ac16d8926bd` | none |
| LIBERO RoboSuite 1.4.0 | `https://github.com/Max-Fu/robosuite` | `a498b087d4bc5a3981e3d27030d09bc537a537f3` | destructor safety patch below |
| SAM3 | repository recorded in bundled source | `6fe87d64a5beb9084923d7a9e002741178635b09` | none |
| PyRoKi | `https://github.com/chungmin99/pyroki.git` | `95afccc22658c461ab1042a048ae4e9c24bc2a47` | installed dependency, not vendored |
| jaxls | `https://github.com/brentyi/jaxls.git` | `6fe7cf9d56223736b2c0272d080ec7346203c648` | installed dependency, not vendored |

The LIBERO RoboSuite patch is stored at
`third_party_patches/libero_robosuite_binding_utils.patch`. It only guards
cleanup of partially initialized OpenGL contexts.

