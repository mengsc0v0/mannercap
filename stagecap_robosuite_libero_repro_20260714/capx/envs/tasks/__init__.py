from .base import (
    CodeExecEnvConfig,
    CodeExecutionEnvBase,
    get_config,
    get_exec_env,
    list_configs,
    list_exec_envs,
    register_config,
    register_exec_env,
)


try:
    from .dclaw.robel_dclaw_env import DClawCodeEnv

    register_exec_env("robel_dclaw_code_env", DClawCodeEnv)
    register_config(
        "robel_dclaw_code_env",
        CodeExecEnvConfig(
            low_level="robel_dclaw_screw_fixed_low_level",
            apis=["DClawControlApi"],
            privileged=False,
        ),
    )
except ImportError:
    print("ROBEL DClaw code environment not installed, skipping DClaw task envs")


try:
    from .allegro_kuka.isaacgym_allegro_kuka_env import AllegroKukaCodeEnv

    register_exec_env("isaacgym_allegro_kuka_code_env", AllegroKukaCodeEnv)
    register_config(
        "isaacgym_allegro_kuka_code_env",
        CodeExecEnvConfig(
            low_level="isaacgym_allegro_kuka_low_level",
            apis=["AllegroKukaControlApi"],
            privileged=False,
            enable_render=False,
        ),
    )
except ImportError:
    print("IsaacGym AllegroKuka code environment not installed, skipping AllegroKuka task envs")


try:
    from .mfr_screwdriver.mfr_screwdriver_env import MfrScrewdriverCodeEnv

    register_exec_env("mfr_screwdriver_code_env", MfrScrewdriverCodeEnv)
    register_config(
        "mfr_screwdriver_code_env",
        CodeExecEnvConfig(
            low_level="mfr_screwdriver_turning_low_level",
            apis=["MfrScrewdriverControlApi"],
            privileged=False,
            enable_render=False,
        ),
    )
except ImportError:
    print("MFR screwdriver code environment not installed, skipping MFR screwdriver task envs")


try:
    from .dexscrew_xhand.dexscrew_xhand_env import DexScrewXHandNutBoltCodeEnv

    register_exec_env("dexscrew_xhand_nutbolt_code_env", DexScrewXHandNutBoltCodeEnv)
    register_config(
        "dexscrew_xhand_nutbolt_code_env",
        CodeExecEnvConfig(
            low_level="dexscrew_xhand_nutbolt_low_level",
            apis=["DexScrewXHandNutBoltControlApi"],
            privileged=False,
            enable_render=False,
        ),
    )
except ImportError:
    print("DexScrew XHand code environment not installed, skipping DexScrew task envs")


try:
    from .franka.franka_cube_restack import FrankaRestackCodeEnv
    from .franka.franka_lift import FrankaLiftCodeEnv
    from .franka.franka_nut_assembly import FrankaNutAssemblyCodeEnv
    from .franka.franka_pick_place import FrankaPickPlaceCodeEnv
    from .franka.franka_spill_wipe import FrankaSpillWipeCodeEnv
    from .franka.two_arm_handover import TwoArmHandoverCodeEnv
    from .franka.two_arm_lift import TwoArmLiftCodeEnv

    register_exec_env("franka_real_code_env", FrankaPickPlaceCodeEnv)
    register_config(
        "franka_real_code_env",
        CodeExecEnvConfig(
            low_level="franka_real_low_level",
            apis=["FrankaControlApi"],
        ),
    )
    register_exec_env("franka_robosuite_spill_wipe_code_env", FrankaSpillWipeCodeEnv)
    register_config(
        "franka_robosuite_spill_wipe_code_env",
        CodeExecEnvConfig(
            low_level="franka_robosuite_spill_wipe_low_level",
            apis=["FrankaControlSpillWipePrivilegedApi"],
        ),
    )

    register_exec_env("franka_pick_place_code_env", FrankaPickPlaceCodeEnv)
    register_config(
        "franka_pick_place_code_env",
        CodeExecEnvConfig(
            low_level="franka_cubes_low_level",
            apis=["FrankaControlPrivilegedApi"],
        ),
    )
    register_exec_env("franka_robosuite_pick_place_code_env", FrankaPickPlaceCodeEnv)
    register_config(
        "franka_robosuite_pick_place_code_env",
        CodeExecEnvConfig(
            low_level="franka_robosuite_cubes_low_level",
            apis=["FrankaControlPrivilegedApi"],
        ),
    )
    register_exec_env("franka_nut_assembly_code_env", FrankaNutAssemblyCodeEnv)
    register_config(
        "franka_nut_assembly_code_env",
        CodeExecEnvConfig(
            low_level="franka_robosuite_nut_assembly_low_level",
            apis=["FrankaControlNutAssemblyPrivilegedApi"],
            privileged=True,
        ),
    )

    register_exec_env("franka_nut_assembly_code_env_visual", FrankaNutAssemblyCodeEnv)
    register_config(
        "franka_nut_assembly_code_env_visual",
        CodeExecEnvConfig(
            low_level="franka_robosuite_nut_assembly_low_level_visual",
            apis=["FrankaControlNutAssemblyVisualApi"],
            privileged=False,
        ),
    )
    register_exec_env("franka_pick_place_multi_code_env", FrankaPickPlaceCodeEnv)
    register_config(
        "franka_pick_place_multi_code_env",
        CodeExecEnvConfig(
            low_level="franka_cubes_low_level",
            apis=["FrankaControlMultiPrivilegedApi"],
        ),
    )

    register_exec_env("franka_lift_code_env", FrankaLiftCodeEnv)
    register_config(
        "franka_lift_code_env",
        CodeExecEnvConfig(
            low_level="franka_robosuite_cube_lift_low_level",
            apis=["FrankaControlPrivilegedApi"],
        ),
    )
    register_exec_env("two_arm_handover_code_env", TwoArmHandoverCodeEnv)
    register_config(
        "two_arm_handover_code_env",
        CodeExecEnvConfig(
            low_level="two_arm_handover_robosuite",
            apis=["FrankaHandoverApi"],
        ),
    )

    register_exec_env("franka_restack_code_env", FrankaRestackCodeEnv)
    register_config(
        "franka_restack_code_env",
        CodeExecEnvConfig(
            low_level="franka_robosuite_cubes_restack_low_level",
            apis=["FrankaControlPrivilegedApi"],
        ),
    )
except ImportError:
    print("Franka code environments not fully installed, skipping Franka task envs")


try:
    from .franka.franka_libero_env import FrankaLiberoCodeEnv

    register_exec_env("franka_libero_code_env", FrankaLiberoCodeEnv)
    register_config(
        "franka_libero_code_env",
        CodeExecEnvConfig(
            low_level="franka_libero_low_level",
            apis=["FrankaLiberoApi"],
        ),
    )
except ImportError:
    print("LIBERO code environment not installed, skipping LIBERO task envs")


try:
    from .r1pro.r1pro_pickup_radio import R1ProRadioCodeEnv
    from .r1pro.r1pro_pickup_trash import R1ProTrashCodeEnv

    register_exec_env("r1pro_radio_code_env", R1ProRadioCodeEnv)
    register_config(
        "r1pro_radio_code_env",
        CodeExecEnvConfig(
            low_level="r1pro_b1k_low_level",
            apis=["R1ProControlApi"],
        ),
    )
    register_exec_env("r1pro_trash_code_env", R1ProTrashCodeEnv)
    register_config(
        "r1pro_trash_code_env",
        CodeExecEnvConfig(
            low_level="r1pro_b1k_low_level",
            apis=["R1ProControlApi"],
        ),
    )
except ImportError:
    print("R1Pro code environments not installed, skipping R1Pro task envs")
